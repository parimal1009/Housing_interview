"""
Housing System Interview Application
A professional AI-powered housing assessment tool with audio transcription and analysis.
"""
import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import uuid
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Audio processing imports
import wave

# AI/ML imports
from groq import Groq
from langchain_groq import ChatGroq
from langsmith import Client

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
MAX_AUDIO_SIZE_MB = 25
SUPPORTED_AUDIO_FORMATS = {'.mp3', '.mp4', '.wav', '.m4a', '.webm', '.ogg'}
DEFAULT_PORT = 8001
TRANSCRIPTION_TIMEOUT = 300  # 5 minutes

# Pydantic models for request/response validation
class TranscriptionResponse(BaseModel):
    raw_text: str
    cleaned_text: str

class QuestionResponse(BaseModel):
    question: Dict[str, Any]
    question_number: int
    total_questions: int
    progress: float

class AnalysisResponse(BaseModel):
    analysis: str
    timestamp: str
    question_id: int
    session_id: str

class SessionResponse(BaseModel):
    session_id: str
    total_questions: int

class HealthResponse(BaseModel):
    status: str
    transcription_method: str
    models_loaded: Dict[str, bool]
    timestamp: str

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("🚀 Starting Housing Interview Application...")
    # Startup: Initialize models
    global ai_models
    ai_models = AIModels()
    logger.info("✅ Application started successfully")
    yield
    # Shutdown: Cleanup
    logger.info("🛑 Shutting down application...")
    if ai_models and hasattr(ai_models, 'cleanup'):
        ai_models.cleanup()

# Initialize FastAPI app with enhanced configuration
app = FastAPI(
    title="Housing System Interview Application",
    description="AI-powered housing assessment tool with audio transcription and analysis",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize AI models and clients
class AIModels:
    """Manages all AI/ML models with proper initialization and error handling"""
    
    def __init__(self):
        self.transcription_method: Optional[Dict[str, Any]] = None
        self.groq_client: Optional[Groq] = None
        self.langchain_llm: Optional[ChatGroq] = None
        self.langsmith_client: Optional[Client] = None
        self._initialized = False
        self.initialize_models()
    
    @property
    def is_ready(self) -> bool:
        """Check if models are initialized and ready"""
        return self._initialized and self.transcription_method is not None
    
    def initialize_models(self) -> None:
        """Initialize all AI models and clients with fallback options"""
        try:
            # Initialize transcription (try multiple options)
            self.transcription_method = self._setup_transcription()
            
            # Initialize Groq client
            groq_api_key = os.getenv("GROQ_API_KEY")
            if groq_api_key:
                try:
                    self.groq_client = Groq(api_key=groq_api_key)
                    logger.info("✅ Groq client initialized successfully!")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize Groq client: {e}")
            else:
                logger.warning("⚠️ No GROQ_API_KEY found - AI analysis will be limited")
            
            # Initialize LangChain with Groq
            if groq_api_key:
                try:
                    self.langchain_llm = ChatGroq(
                        groq_api_key=groq_api_key,
                        model_name="llama-3.3-70b-versatile",
                        temperature=0.7,
                        max_retries=3,
                        timeout=30.0
                    )
                    logger.info("✅ LangChain with Groq initialized successfully!")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize LangChain: {e}")
            
            # Initialize LangSmith client
            langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
            if langsmith_api_key:
                try:
                    self.langsmith_client = Client(
                        api_url=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
                        api_key=langsmith_api_key
                    )
                    logger.info("✅ LangSmith client initialized successfully!")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize LangSmith: {e}")
            
            self._initialized = True
            logger.info("✅ All models initialized")
            
        except Exception as e:
            logger.error(f"❌ Critical error initializing models: {str(e)}")
            self._initialized = False
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        logger.info("Cleaning up AI models...")
        # Add any cleanup logic here if needed
    
    def _setup_transcription(self) -> Dict[str, Any]:
        """Setup transcription with multiple fallback options"""
        transcription_engines = [
            ("faster_whisper", self._init_faster_whisper, "Faster-Whisper (recommended)"),
            ("vosk", self._init_vosk, "Vosk (lightweight)"),
            ("openai_whisper", self._init_openai_whisper, "OpenAI Whisper"),
            ("speech_recognition", self._init_speech_recognition, "SpeechRecognition"),
        ]
        
        for method_name, init_func, display_name in transcription_engines:
            try:
                model = init_func()
                if model is not None:
                    logger.info(f"✅ {display_name} initialized successfully!")
                    return {"method": method_name, "model": model}
            except ImportError:
                logger.debug(f"{display_name} not installed, trying next option...")
            except Exception as e:
                logger.warning(f"{display_name} initialization failed: {e}")
        
        # Fallback: Mock transcription (for development)
        logger.warning("⚠️ Using mock transcription - install a transcription library!")
        return {"method": "mock", "model": None}
    
    def _init_faster_whisper(self):
        """Initialize Faster-Whisper"""
        from faster_whisper import WhisperModel
        return WhisperModel("base", device="cpu", compute_type="int8")
    
    def _init_vosk(self):
        """Initialize Vosk"""
        import vosk
        model_path = self._ensure_vosk_model()
        if model_path:
            return vosk.Model(model_path)
        return None
    
    def _init_openai_whisper(self):
        """Initialize OpenAI Whisper"""
        import whisper
        return whisper.load_model("base")
    
    def _init_speech_recognition(self):
        """Initialize SpeechRecognition"""
        import speech_recognition as sr
        return sr.Recognizer()
    
    def _ensure_vosk_model(self) -> Optional[str]:
        """Download Vosk model if not present"""
        model_dir = Path("vosk-model")
        
        if model_dir.exists():
            return str(model_dir)
        
        try:
            import requests
            import zipfile
            
            model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
            zip_path = Path("vosk-model.zip")
            
            logger.info("Downloading Vosk model...")
            response = requests.get(model_url, timeout=60)
            response.raise_for_status()
            
            zip_path.write_bytes(response.content)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            # Rename extracted folder
            extracted_name = Path("vosk-model-small-en-us-0.15")
            if extracted_name.exists():
                extracted_name.rename(model_dir)
            
            zip_path.unlink()
            logger.info("✅ Vosk model downloaded successfully!")
            return str(model_dir)
            
        except Exception as e:
            logger.error(f"❌ Failed to download Vosk model: {e}")
            return None

# Initialize models globally
ai_models = AIModels()

class HousingInterviewer:
    """Main class for housing system interviews with improved session management"""
    
    def __init__(self):
        self.questions: List[Dict[str, Any]] = self._get_housing_questions()
        self.interview_sessions: Dict[str, Dict[str, Any]] = {}
        self._session_lock = asyncio.Lock()
    
    async def create_session(self) -> str:
        """Create a new interview session with thread-safe ID generation"""
        async with self._session_lock:
            session_id = str(uuid.uuid4())
            self.interview_sessions[session_id] = {
                "created_at": datetime.now().isoformat(),
                "current_question": 0,
                "responses": [],
                "participant_info": {},
                "last_activity": datetime.now().isoformat()
            }
            return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data with validation"""
        return self.interview_sessions.get(session_id)
    
    def update_session_activity(self, session_id: str) -> None:
        """Update last activity timestamp"""
        if session_id in self.interview_sessions:
            self.interview_sessions[session_id]["last_activity"] = datetime.now().isoformat()
    
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
        """Transcribe audio using available transcription method with validation"""
        # Validate file size
        audio_bytes = await audio_file.read()
        file_size_mb = len(audio_bytes) / (1024 * 1024)
        
        if file_size_mb > MAX_AUDIO_SIZE_MB:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({MAX_AUDIO_SIZE_MB}MB)"
            )
        
        # Validate file format
        file_ext = Path(audio_file.filename).suffix.lower()
        if file_ext not in SUPPORTED_AUDIO_FORMATS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported audio format: {file_ext}. Supported: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
            )
        
        tmp_path = None
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
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
            if tmp_path and Path(tmp_path).exists():
                try:
                    Path(tmp_path).unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {tmp_path}: {e}")
            
            # Process and clean the transcription
            cleaned_transcription = await self._clean_transcription(raw_transcription)
            
            return {
                "raw_text": raw_transcription,
                "cleaned_text": cleaned_transcription
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Transcription failed: {str(e)}"
            )
        finally:
            # Ensure cleanup
            if tmp_path and Path(tmp_path).exists():
                try:
                    Path(tmp_path).unlink()
                except:
                    pass
    
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
        """Clean and process transcription using LLM with timeout and retry"""
        if not raw_text or not raw_text.strip():
            return raw_text
        
        if not ai_models.groq_client:
            logger.debug("Groq client not available, returning raw text")
            return raw_text
        
        try:
            
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
                max_tokens=1000,
                timeout=30
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning transcription: {e}", exc_info=True)
            return raw_text  # Graceful fallback
    
    async def analyze_response(self, question: Dict, transcription: str, session_id: str) -> Dict[str, Any]:
        """Analyze interview response using LLM with comprehensive error handling"""
        timestamp = datetime.now().isoformat()
        
        if not ai_models.groq_client:
            logger.warning("Groq client not available for analysis")
            return {
                "analysis": "Response recorded successfully. AI analysis temporarily unavailable.",
                "timestamp": timestamp,
                "question_id": question['id'],
                "session_id": session_id
            }
        
        try:
            
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
                max_tokens=800,
                timeout=30
            )
            
            analysis = response.choices[0].message.content.strip()
            
            # Log to LangSmith if available (non-blocking)
            if ai_models.langsmith_client:
                asyncio.create_task(self._log_to_langsmith(question, transcription, analysis, session_id))
            
            return {
                "analysis": analysis,
                "timestamp": timestamp,
                "question_id": question['id'],
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error analyzing response: {e}", exc_info=True)
            return {
                "analysis": "Response recorded successfully. Analysis temporarily unavailable due to technical issues.",
                "error": str(e),
                "timestamp": timestamp,
                "question_id": question.get('id', 0),
                "session_id": session_id
            }
    
    async def _log_to_langsmith(self, question: Dict, transcription: str, analysis: str, session_id: str) -> None:
        """Log analysis to LangSmith asynchronously"""
        try:
            ai_models.langsmith_client.create_run(
                name="housing_response_analysis",
                run_type="llm",
                inputs={
                    "question_category": question.get('category', 'Unknown'),
                    "question": question.get('question', ''),
                    "response": transcription,
                    "session_id": session_id
                },
                outputs={"analysis": analysis},
                project_name=os.getenv("LANGSMITH_PROJECT", "HOUSING_SYSTEM")
            )
        except Exception as e:
            logger.error(f"LangSmith logging error: {e}")
    
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
async def read_root(request: Request) -> HTMLResponse:
    """Main page with proper error handling"""
    template_path = Path("templates/index.html")
    
    try:
        if not template_path.exists():
            logger.error(f"Template not found: {template_path}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Application template not found"
            )
        
        html_content = template_path.read_text(encoding="utf-8")
        return HTMLResponse(content=html_content)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading template: {e}")
        # Fallback HTML
        return HTMLResponse(
            content="""
            <!DOCTYPE html>
            <html>
            <head><title>Housing Interview - Error</title></head>
            <body>
                <h1>Housing System Interview</h1>
                <p>Application error. Please contact support.</p>
            </body>
            </html>
            """,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@app.post("/start_session", response_model=SessionResponse)
async def start_interview_session() -> SessionResponse:
    """Start a new interview session with validation"""
    try:
        session_id = await interviewer.create_session()
        return SessionResponse(
            session_id=session_id,
            total_questions=len(interviewer.questions)
        )
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create interview session"
        )

@app.get("/get_question/{session_id}")
async def get_question(session_id: str, question_num: Optional[int] = None) -> Dict[str, Any]:
    """Get current or specific question with validation"""
    session = interviewer.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    interviewer.update_session_activity(session_id)
    
    if question_num is None:
        question_num = session.get("current_question", 0)
    
    if question_num < 0 or question_num >= len(interviewer.questions):
        if question_num >= len(interviewer.questions):
            return {"completed": True, "message": "Interview completed"}
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid question number"
        )
    
    question = interviewer.questions[question_num]
    return {
        "question": question,
        "question_number": question_num + 1,
        "total_questions": len(interviewer.questions),
        "progress": ((question_num + 1) / len(interviewer.questions)) * 100
    }

@app.post("/transcribe_audio/{session_id}")
async def transcribe_audio_endpoint(
    session_id: str,
    audio_file: UploadFile = File(...)
) -> Dict[str, Any]:
    """Transcribe uploaded audio file with comprehensive validation"""
    # Validate session
    session = interviewer.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    interviewer.update_session_activity(session_id)
    
    # Validate filename
    if not audio_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided"
        )
    
    try:
        # Transcribe audio
        transcription_result = await interviewer.transcribe_audio(audio_file)
        
        return {
            "transcription": transcription_result,
            "filename": audio_file.filename,
            "session_id": session_id,
            "transcription_method": ai_models.transcription_method.get("method", "unknown") if ai_models.transcription_method else "none"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )

@app.post("/submit_response/{session_id}")
async def submit_response(
    session_id: str,
    question_id: int = Form(..., ge=1),
    raw_transcription: str = Form(...),
    cleaned_transcription: str = Form(...)
) -> Dict[str, Any]:
    """Submit interview response for analysis with validation"""
    # Validate session
    session = interviewer.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    interviewer.update_session_activity(session_id)
    
    # Validate question ID
    if question_id < 1 or question_id > len(interviewer.questions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid question ID: {question_id}"
        )
    
    # Validate transcription
    if not cleaned_transcription.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transcription cannot be empty"
        )
    
    try:
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Submit response error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit response: {str(e)}"
        )

@app.get("/generate_report/{session_id}")
async def generate_report(session_id: str) -> Dict[str, Any]:
    """Generate final interview report with validation"""
    session = interviewer.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    interviewer.update_session_activity(session_id)
    
    try:
        report = await interviewer.generate_summary_report(session_id)
        return report
        
    except Exception as e:
        logger.error(f"Generate report error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate report: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint with detailed status"""
    transcription_status = "none"
    if ai_models.transcription_method:
        transcription_status = ai_models.transcription_method.get("method", "unknown")
    
    return HealthResponse(
        status="healthy" if ai_models.is_ready else "degraded",
        transcription_method=transcription_status,
        models_loaded={
            "transcription": ai_models.transcription_method is not None,
            "groq": ai_models.groq_client is not None,
            "langchain": ai_models.langchain_llm is not None,
            "langsmith": ai_models.langsmith_client is not None
        },
        timestamp=datetime.now().isoformat()
    )

def print_startup_info():
    """Print application startup information"""
    print("\n" + "=" * 60)
    print("🏠 HOUSING SYSTEM INTERVIEW APPLICATION")
    print("=" * 60)
    
    # Print model status
    if ai_models.transcription_method:
        method = ai_models.transcription_method.get("method", "unknown")
        print(f"✅ Transcription: {method}")
    else:
        print("❌ Transcription: Not available")
    
    print(f"✅ Groq API: {'Available' if ai_models.groq_client else 'Not configured'}")
    print(f"✅ LangChain: {'Available' if ai_models.langchain_llm else 'Not configured'}")
    print(f"✅ LangSmith: {'Available' if ai_models.langsmith_client else 'Not configured'}")
    
    print("\n" + "=" * 60)
    print("📦 INSTALLATION TIPS")
    print("=" * 60)
    print("For transcription support, install one of:")
    print("  • pip install faster-whisper  (Recommended)")
    print("  • pip install vosk  (Lightweight)")
    print("  • pip install openai-whisper  (Original)")
    print("  • pip install speechrecognition  (Basic)")
    
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    print("\n" + "=" * 60)
    print(f"🌐 Server starting on http://localhost:{port}")
    print(f"📚 API Docs: http://localhost:{port}/api/docs")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    
    print_startup_info()
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info",
        access_log=True
    )
