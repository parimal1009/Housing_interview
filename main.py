import os
import io
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import tempfile
import uuid
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Audio processing imports
import librosa
import numpy as np
import subprocess
import wave

# AI/ML imports
from groq import Groq
from langchain.schema import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langsmith import Client
import re

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Housing System Interview Application", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize AI models and clients
class AIModels:
    def __init__(self):
        self.transcription_method = None
        self.groq_client = None
        self.langchain_llm = None
        self.langsmith_client = None
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all AI models and clients with fallback options"""
        try:
            # Initialize transcription (try multiple options)
            self.transcription_method = self._setup_transcription()
            
            # Initialize Groq client
            if os.getenv("GROQ_API_KEY"):
                self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                logger.info("✅ Groq client initialized successfully!")
            else:
                logger.warning("⚠️ No GROQ_API_KEY found")
            
            # Initialize LangChain with Groq
            if os.getenv("GROQ_API_KEY"):
                self.langchain_llm = ChatGroq(
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.7
                )
                logger.info("✅ LangChain with Groq initialized successfully!")
            
            # Initialize LangSmith client
            if os.getenv("LANGSMITH_API_KEY"):
                self.langsmith_client = Client(
                    api_url=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
                    api_key=os.getenv("LANGSMITH_API_KEY")
                )
                logger.info("✅ LangSmith client initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error initializing models: {str(e)}")
    
    def _setup_transcription(self):
        """Setup transcription with multiple fallback options"""
        # Option 1: Try Faster-Whisper (most efficient)
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("base", device="cpu", compute_type="int8")
            logger.info("✅ Faster-Whisper initialized successfully!")
            return {"method": "faster_whisper", "model": model}
        except ImportError:
            logger.info("Faster-Whisper not available, trying Vosk...")
        except Exception as e:
            logger.warning(f"Faster-Whisper failed: {e}, trying Vosk...")
        
        # Option 2: Try Vosk (lightweight, offline)
        try:
            import vosk
            import json
            # Download model if needed
            model_path = self._ensure_vosk_model()
            if model_path:
                vosk_model = vosk.Model(model_path)
                logger.info("✅ Vosk initialized successfully!")
                return {"method": "vosk", "model": vosk_model}
        except ImportError:
            logger.info("Vosk not available, trying OpenAI Whisper...")
        except Exception as e:
            logger.warning(f"Vosk failed: {e}, trying OpenAI Whisper...")
        
        # Option 3: Try OpenAI Whisper (original)
        try:
            import whisper
            model = whisper.load_model("base")
            logger.info("✅ OpenAI Whisper initialized successfully!")
            return {"method": "openai_whisper", "model": model}
        except ImportError:
            logger.info("OpenAI Whisper not available, trying SpeechRecognition...")
        except Exception as e:
            logger.warning(f"OpenAI Whisper failed: {e}, trying SpeechRecognition...")
        
        # Option 4: Try SpeechRecognition with Google (requires internet)
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            logger.info("✅ SpeechRecognition initialized successfully!")
            return {"method": "speech_recognition", "model": recognizer}
        except ImportError:
            logger.error("No transcription libraries available!")
        except Exception as e:
            logger.error(f"SpeechRecognition failed: {e}")
        
        # Fallback: Mock transcription (for development)
        logger.warning("⚠️ Using mock transcription - install a transcription library!")
        return {"method": "mock", "model": None}
    
    def _ensure_vosk_model(self):
        """Download Vosk model if not present"""
        model_dir = "vosk-model"
        if not os.path.exists(model_dir):
            try:
                import requests
                import zipfile
                
                model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
                logger.info("Downloading Vosk model...")
                
                response = requests.get(model_url)
                with open("vosk-model.zip", "wb") as f:
                    f.write(response.content)
                
                with zipfile.ZipFile("vosk-model.zip", 'r') as zip_ref:
                    zip_ref.extractall(".")
                
                # Rename extracted folder
                extracted_name = "vosk-model-small-en-us-0.15"
                if os.path.exists(extracted_name):
                    os.rename(extracted_name, model_dir)
                
                os.remove("vosk-model.zip")
                logger.info("Vosk model downloaded successfully!")
                return model_dir
            except Exception as e:
                logger.error(f"Failed to download Vosk model: {e}")
                return None
        return model_dir

# Initialize models globally
ai_models = AIModels()

class HousingInterviewer:
    """Main class for housing system interviews"""
    
    def __init__(self):
        self.questions = self._get_housing_questions()
        self.interview_sessions = {}
    
    def _get_housing_questions(self) -> List[Dict[str, Any]]:
        """Define comprehensive housing interview questions"""
        return [
            {
                "id": 1,
                "category": "Personal Information",
                "question": "Can you please state your full name, age, and how long you've been in the current housing situation?",
                "type": "open_ended",
                "importance": "high"
            },
            {
                "id": 2,
                "category": "Housing Conditions",
                "question": "How would you describe the physical condition of your current housing? Are there any maintenance issues or safety concerns?",
                "type": "descriptive",
                "importance": "high"
            },
            {
                "id": 3,
                "category": "Food Security",
                "question": "Tell me about your access to food. Do you have adequate kitchen facilities? How often are you able to have nutritious meals?",
                "type": "assessment",
                "importance": "high"
            },
            {
                "id": 4,
                "category": "Healthcare Access",
                "question": "Do you have access to healthcare services? How do you manage medical appointments and medications?",
                "type": "access_evaluation",
                "importance": "high"
            },
            {
                "id": 5,
                "category": "Community Integration",
                "question": "How do you feel about your connection to the local community? Do you participate in community activities or have social support?",
                "type": "social_assessment",
                "importance": "medium"
            },
            {
                "id": 6,
                "category": "Transportation",
                "question": "What are your transportation options? Can you easily access work, shopping, and services?",
                "type": "accessibility",
                "importance": "medium"
            },
            {
                "id": 7,
                "category": "Financial Situation",
                "question": "How manageable are your housing costs? Do you receive any housing assistance or support?",
                "type": "financial_assessment",
                "importance": "high"
            },
            {
                "id": 8,
                "category": "Safety and Security",
                "question": "Do you feel safe in your current housing situation? Are there any security concerns in your neighborhood?",
                "type": "safety_evaluation",
                "importance": "high"
            },
            {
                "id": 9,
                "category": "Future Plans",
                "question": "What are your housing goals for the future? What kind of support would be most helpful?",
                "type": "planning",
                "importance": "medium"
            },
            {
                "id": 10,
                "category": "Overall Experience",
                "question": "Is there anything else about your housing experience that you'd like to share? Any recommendations for improving housing services?",
                "type": "open_feedback",
                "importance": "medium"
            }
        ]
    
    async def transcribe_audio(self, audio_file: UploadFile) -> Dict[str, str]:
        """Transcribe audio using available transcription method"""
        try:
            # Read audio file
            audio_bytes = await audio_file.read()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            transcription_method = ai_models.transcription_method
            
            if not transcription_method:
                raise Exception("No transcription method available")
            
            method = transcription_method["method"]
            model = transcription_method["model"]
            
            raw_transcription = ""
            
            if method == "faster_whisper":
                raw_transcription = await self._transcribe_faster_whisper(model, tmp_path)
            elif method == "vosk":
                raw_transcription = await self._transcribe_vosk(model, tmp_path)
            elif method == "openai_whisper":
                raw_transcription = await self._transcribe_openai_whisper(model, tmp_path)
            elif method == "speech_recognition":
                raw_transcription = await self._transcribe_speech_recognition(model, tmp_path)
            elif method == "mock":
                raw_transcription = "This is a mock transcription for testing purposes. Please install a transcription library for actual functionality."
            
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            # Process and clean the transcription
            cleaned_transcription = await self._clean_transcription(raw_transcription)
            
            return {
                "raw_text": raw_transcription,
                "cleaned_text": cleaned_transcription
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    async def _transcribe_faster_whisper(self, model, audio_path):
        """Transcribe using Faster-Whisper"""
        try:
            segments, info = model.transcribe(audio_path, beam_size=5)
            return " ".join([segment.text for segment in segments])
        except Exception as e:
            raise Exception(f"Faster-Whisper transcription failed: {str(e)}")
    
    async def _transcribe_vosk(self, model, audio_path):
        """Transcribe using Vosk"""
        try:
            import vosk
            import json
            
            # Convert to proper format for Vosk
            wf = wave.open(audio_path, 'rb')
            rec = vosk.KaldiRecognizer(model, wf.getframerate())
            
            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    results.append(result.get('text', ''))
            
            final_result = json.loads(rec.FinalResult())
            results.append(final_result.get('text', ''))
            
            return " ".join(results).strip()
        except Exception as e:
            raise Exception(f"Vosk transcription failed: {str(e)}")
    
    async def _transcribe_openai_whisper(self, model, audio_path):
        """Transcribe using OpenAI Whisper"""
        try:
            result = model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            raise Exception(f"OpenAI Whisper transcription failed: {str(e)}")
    
    async def _transcribe_speech_recognition(self, recognizer, audio_path):
        """Transcribe using SpeechRecognition"""
        try:
            import speech_recognition as sr
            
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
            
            # Try Google Speech Recognition (requires internet)
            return recognizer.recognize_google(audio)
        except Exception as e:
            raise Exception(f"SpeechRecognition transcription failed: {str(e)}")
    
    async def _clean_transcription(self, raw_text: str) -> str:
        """Clean and process transcription using LLM"""
        try:
            if not ai_models.groq_client or not raw_text.strip():
                return raw_text
            
            cleaning_prompt = f"""
            You are an expert transcription cleaner. Clean the following transcription by:
            1. Correcting obvious speech-to-text errors
            2. Adding proper punctuation and capitalization
            3. Removing filler words (um, uh, like) appropriately
            4. Maintaining the original meaning and tone
            5. Organizing into clear sentences
            
            Raw transcription: {raw_text}
            
            Return only the cleaned transcription, nothing else.
            """
            
            response = ai_models.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a professional transcription editor."},
                    {"role": "user", "content": cleaning_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning transcription: {e}")
            return raw_text  # Return raw text if cleaning fails
    
    async def analyze_response(self, question: Dict, transcription: str, session_id: str) -> Dict[str, Any]:
        """Analyze interview response using LLM"""
        try:
            if not ai_models.groq_client:
                return {
                    "analysis": "Response recorded successfully. Analysis temporarily unavailable.",
                    "timestamp": datetime.now().isoformat(),
                    "question_id": question['id'],
                    "session_id": session_id
                }
            
            analysis_prompt = f"""
            You are a professional housing services analyst. Analyze this interview response:
            
            Question Category: {question['category']}
            Question: {question['question']}
            Response: {transcription}
            
            Provide analysis in the following format:
            1. Key Points: [Main points from the response]
            2. Concerns: [Any concerns or red flags]
            3. Follow-up Needed: [Whether follow-up is recommended]
            4. Sentiment: [Positive/Neutral/Negative]
            5. Support Recommendations: [Suggested support services]
            
            Be thorough but concise.
            """
            
            response = ai_models.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert in housing services and social work analysis."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.5,
                max_tokens=800
            )
            
            analysis = response.choices[0].message.content.strip()
            
            # Log to LangSmith if available
            if ai_models.langsmith_client:
                try:
                    ai_models.langsmith_client.create_run(
                        name="housing_response_analysis",
                        run_type="llm",
                        inputs={
                            "question_category": question['category'],
                            "question": question['question'],
                            "response": transcription,
                            "session_id": session_id
                        },
                        outputs={"analysis": analysis},
                        project_name=os.getenv("LANGSMITH_PROJECT", "HOUSING_SYSTEM")
                    )
                except Exception as logging_error:
                    logger.error(f"LangSmith logging error: {logging_error}")
            
            return {
                "analysis": analysis,
                "timestamp": datetime.now().isoformat(),
                "question_id": question['id'],
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error analyzing response: {e}")
            return {
                "analysis": "Response recorded successfully. Analysis temporarily unavailable due to technical issues.",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def generate_summary_report(self, session_id: str) -> Dict[str, Any]:
        """Generate comprehensive interview summary"""
        try:
            session_data = self.interview_sessions.get(session_id, {})
            responses = session_data.get('responses', [])
            
            if not responses:
                return {"error": "No responses found for this session"}
            
            if not ai_models.groq_client:
                # Generate basic summary without LLM
                basic_summary = f"""
HOUSING INTERVIEW SUMMARY REPORT

Interview Date: {session_data.get('created_at', 'Unknown')}
Session ID: {session_id}
Total Responses: {len(responses)}

RESPONSES SUMMARY:
"""
                for i, response in enumerate(responses, 1):
                    basic_summary += f"\n{i}. {response['category']}: {response.get('transcription', 'No response')[:100]}..."
                
                return {
                    "summary_report": basic_summary,
                    "session_id": session_id,
                    "total_responses": len(responses),
                    "generated_at": datetime.now().isoformat(),
                    "participant_info": session_data.get('participant_info', {})
                }
            
            # Compile all responses for analysis
            compiled_responses = "\n\n".join([
                f"Q{r['question_id']}: {r['question']}\nA: {r['transcription']}\nAnalysis: {r.get('analysis', 'N/A')}"
                for r in responses
            ])
            
            summary_prompt = f"""
            Based on this comprehensive housing interview, create a detailed summary report:
            
            {compiled_responses}
            
            Please provide:
            1. OVERALL ASSESSMENT: General housing situation summary
            2. KEY STRENGTHS: Positive aspects of current situation
            3. PRIORITY CONCERNS: Most urgent issues requiring attention
            4. RECOMMENDATIONS: Specific actionable recommendations
            5. RESOURCE NEEDS: Required support services and resources
            6. FOLLOW-UP ACTIONS: Next steps for case management
            
            Format as a professional social services report.
            """
            
            response = ai_models.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a senior housing services coordinator creating official reports."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.4,
                max_tokens=1500
            )
            
            summary = response.choices[0].message.content.strip()
            
            return {
                "summary_report": summary,
                "session_id": session_id,
                "total_responses": len(responses),
                "generated_at": datetime.now().isoformat(),
                "participant_info": session_data.get('participant_info', {})
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {"error": f"Failed to generate summary: {str(e)}"}

# Initialize interviewer
interviewer = HousingInterviewer()

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Main page"""
    # Read the HTML file directly since templates directory might not exist
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        # Fallback HTML if template not found
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head><title>Housing Interview</title></head>
        <body>
            <h1>Housing System Interview</h1>
            <p>Template file not found. Please ensure templates/index.html exists.</p>
        </body>
        </html>
        """)

@app.post("/start_session")
async def start_interview_session():
    """Start a new interview session"""
    session_id = str(uuid.uuid4())
    interviewer.interview_sessions[session_id] = {
        "created_at": datetime.now().isoformat(),
        "current_question": 0,
        "responses": [],
        "participant_info": {}
    }
    return {"session_id": session_id, "total_questions": len(interviewer.questions)}

@app.get("/get_question/{session_id}")
async def get_question(session_id: str, question_num: Optional[int] = None):
    """Get current or specific question"""
    if session_id not in interviewer.interview_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = interviewer.interview_sessions[session_id]
    
    if question_num is None:
        question_num = session.get("current_question", 0)
    
    if question_num >= len(interviewer.questions):
        return {"completed": True, "message": "Interview completed"}
    
    question = interviewer.questions[question_num]
    return {
        "question": question,
        "question_number": question_num + 1,
        "total_questions": len(interviewer.questions),
        "progress": ((question_num + 1) / len(interviewer.questions)) * 100
    }

@app.post("/transcribe_audio/{session_id}")
async def transcribe_audio_endpoint(session_id: str, audio_file: UploadFile = File(...)):
    """Transcribe uploaded audio file"""
    try:
        if session_id not in interviewer.interview_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Validate file type
        allowed_extensions = ['.mp3', '.mp4', '.wav', '.m4a', '.webm', '.ogg']
        if not any(audio_file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(status_code=400, detail="Unsupported audio format")
        
        # Transcribe audio
        transcription_result = await interviewer.transcribe_audio(audio_file)
        
        return {
            "transcription": transcription_result,
            "filename": audio_file.filename,
            "session_id": session_id,
            "transcription_method": ai_models.transcription_method.get("method", "unknown") if ai_models.transcription_method else "none"
        }
        
    except Exception as e:
        logger.error(f"Transcription endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/submit_response/{session_id}")
async def submit_response(
    session_id: str,
    question_id: int = Form(...),
    raw_transcription: str = Form(...),
    cleaned_transcription: str = Form(...)
):
    """Submit interview response for analysis"""
    try:
        if session_id not in interviewer.interview_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = interviewer.interview_sessions[session_id]
        
        # Get question details
        if question_id - 1 >= len(interviewer.questions):
            raise HTTPException(status_code=400, detail="Invalid question ID")
        
        question = interviewer.questions[question_id - 1]
        
        # Analyze the response
        analysis = await interviewer.analyze_response(question, cleaned_transcription, session_id)
        
        # Store response
        response_data = {
            "question_id": question_id,
            "question": question["question"],
            "category": question["category"],
            "raw_transcription": raw_transcription,
            "transcription": cleaned_transcription,
            "analysis": analysis.get("analysis", ""),
            "timestamp": datetime.now().isoformat()
        }
        
        session["responses"].append(response_data)
        session["current_question"] = question_id  # Move to next question
        
        return {
            "success": True,
            "analysis": analysis,
            "next_question": question_id < len(interviewer.questions)
        }
        
    except Exception as e:
        logger.error(f"Submit response error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate_report/{session_id}")
async def generate_report(session_id: str):
    """Generate final interview report"""
    try:
        if session_id not in interviewer.interview_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        report = await interviewer.generate_summary_report(session_id)
        return report
        
    except Exception as e:
        logger.error(f"Generate report error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    transcription_status = "none"
    if ai_models.transcription_method:
        transcription_status = ai_models.transcription_method.get("method", "unknown")
    
    return {
        "status": "healthy",
        "transcription_method": transcription_status,
        "models_loaded": {
            "transcription": ai_models.transcription_method is not None,
            "groq": ai_models.groq_client is not None,
            "langchain": ai_models.langchain_llm is not None
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("🏠 Housing System Interview Application")
    print("=" * 50)
    
    # Print initialization status
    if ai_models.transcription_method:
        method = ai_models.transcription_method.get("method", "unknown")
        print(f"✅ Transcription: {method}")
    else:
        print("❌ Transcription: Not available")
    
    print(f"✅ Groq API: {'Available' if ai_models.groq_client else 'Not configured'}")
    print(f"✅ LangSmith: {'Available' if ai_models.langsmith_client else 'Not configured'}")
    print("=" * 50)
    
    # Installation instructions
    print("\n📦 To install missing dependencies:")
    print("pip install faster-whisper  # Recommended")
    print("pip install vosk  # Lightweight alternative")
    print("pip install openai-whisper  # Original Whisper")
    print("pip install speechrecognition  # Basic option")
    print("\n🌐 Starting server on http://localhost:8000")
    
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    
    print("🏠 Housing System Interview Application")
    print("=" * 50)
    
    # Print initialization status (same as before)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )