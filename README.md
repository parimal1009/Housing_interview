# 🏠 Housing System Interview Application

A professional AI-powered housing assessment tool that conducts structured interviews, transcribes audio responses, and generates comprehensive analysis reports using advanced machine learning models.

## 📋 Overview

This application streamlines the housing assessment process by:
- Conducting structured interviews with 10 comprehensive questions
- Recording and transcribing audio responses using multiple transcription engines
- Analyzing responses using AI (Groq LLM with LangChain)
- Generating detailed summary reports with actionable recommendations
- Providing real-time feedback and insights

## ✨ Key Features

### 🎤 Multi-Engine Audio Transcription
- **Faster-Whisper**: High-performance transcription (recommended)
- **Vosk**: Lightweight offline transcription
- **OpenAI Whisper**: Original Whisper model
- **SpeechRecognition**: Basic Google Speech API fallback
- Automatic fallback system ensures reliability

### 🤖 AI-Powered Analysis
- **Groq API**: Fast LLM inference with Llama 3.3 70B
- **LangChain**: Structured prompt engineering and chain management
- **LangSmith**: Observability and tracing for AI operations
- Automatic transcription cleaning and formatting

### 📊 Comprehensive Assessment
- 10 structured questions covering:
  - Personal Information
  - Housing Conditions
  - Food Security
  - Healthcare Access
  - Community Integration
  - Transportation
  - Financial Situation
  - Safety and Security
  - Future Plans
  - Overall Experience

### 💻 Modern Web Interface
- Responsive design (desktop and mobile)
- Real-time audio recording with timer
- Drag-and-drop file upload
- Dual transcription view (raw + cleaned)
- Progress tracking
- Interactive analysis display

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- FFmpeg (for audio processing)
- API Keys:
  - Groq API Key (required for AI features)
  - LangSmith API Key (optional, for observability)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd housing-interview-app
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install FFmpeg**
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt-get install ffmpeg`

4. **Set up environment variables**

Create a `.env` file in the project root:
```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=HOUSING_SYSTEM
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

5. **Run the application**
```bash
python main.py
```

The application will start on `http://localhost:8001`

## 📦 Installation Options

### Option 1: Faster-Whisper (Recommended)
```bash
pip install faster-whisper
```
Best performance and accuracy.

### Option 2: Vosk (Lightweight)
```bash
pip install vosk
```
Smaller model, works offline, good for resource-constrained environments.

### Option 3: OpenAI Whisper (Original)
```bash
pip install openai-whisper
```
Original Whisper implementation, requires more resources.

### Option 4: SpeechRecognition (Basic)
```bash
pip install SpeechRecognition
```
Simple fallback, requires internet connection.

## 🏗️ Architecture

### Backend (FastAPI)
```
main.py
├── AIModels: Manages all AI/ML models
│   ├── Transcription engines (Faster-Whisper, Vosk, etc.)
│   ├── Groq client (LLM)
│   ├── LangChain integration
│   └── LangSmith observability
│
└── HousingInterviewer: Core interview logic
    ├── Question management
    ├── Audio transcription
    ├── Response analysis
    └── Report generation
```

### Frontend (HTML/CSS/JavaScript)
```
templates/index.html
├── Welcome screen
├── Interview interface
│   ├── Audio recording
│   ├── File upload
│   ├── Transcription display
│   └── Analysis viewer
└── Report generation
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key for LLM inference |
| `LANGSMITH_API_KEY` | No | LangSmith API key for observability |
| `LANGSMITH_PROJECT` | No | LangSmith project name (default: HOUSING_SYSTEM) |
| `LANGSMITH_ENDPOINT` | No | LangSmith API endpoint |
| `PORT` | No | Server port (default: 8001) |

### Transcription Priority

The application tries transcription engines in this order:
1. Faster-Whisper (best performance)
2. Vosk (lightweight)
3. OpenAI Whisper (original)
4. SpeechRecognition (basic)
5. Mock (development fallback)

## 📡 API Endpoints

### Core Endpoints

#### `POST /start_session`
Start a new interview session.

**Response:**
```json
{
  "session_id": "uuid",
  "total_questions": 10
}
```

#### `GET /get_question/{session_id}`
Get current or specific question.

**Query Parameters:**
- `question_num` (optional): Specific question number

**Response:**
```json
{
  "question": {
    "id": 1,
    "category": "Personal Information",
    "question": "Can you please state your full name...",
    "type": "open_ended",
    "importance": "high"
  },
  "question_number": 1,
  "total_questions": 10,
  "progress": 10.0
}
```

#### `POST /transcribe_audio/{session_id}`
Transcribe uploaded audio file.

**Form Data:**
- `audio_file`: Audio file (MP3, WAV, M4A, WebM, OGG)

**Response:**
```json
{
  "transcription": {
    "raw_text": "original transcription",
    "cleaned_text": "cleaned and formatted transcription"
  },
  "filename": "recording.webm",
  "session_id": "uuid",
  "transcription_method": "faster_whisper"
}
```

#### `POST /submit_response/{session_id}`
Submit interview response for analysis.

**Form Data:**
- `question_id`: Question ID
- `raw_transcription`: Raw transcription text
- `cleaned_transcription`: Cleaned transcription text

**Response:**
```json
{
  "success": true,
  "analysis": {
    "analysis": "Detailed analysis text...",
    "timestamp": "2025-12-08T10:30:00",
    "question_id": 1,
    "session_id": "uuid"
  },
  "next_question": true
}
```

#### `GET /generate_report/{session_id}`
Generate final interview summary report.

**Response:**
```json
{
  "summary_report": "Comprehensive report text...",
  "session_id": "uuid",
  "total_responses": 10,
  "generated_at": "2025-12-08T11:00:00",
  "participant_info": {}
}
```

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "transcription_method": "faster_whisper",
  "models_loaded": {
    "transcription": true,
    "groq": true,
    "langchain": true
  },
  "timestamp": "2025-12-08T10:00:00"
}
```

## 🎯 Usage Guide

### Conducting an Interview

1. **Start Interview**: Click "Start Interview" on the welcome screen
2. **Record Response**: 
   - Click "🎤 Start Recording" to record audio
   - Or upload an audio file via drag-and-drop or file picker
3. **Review Transcription**: 
   - View raw transcription (direct from speech-to-text)
   - Edit cleaned transcription if needed
4. **Submit**: Click "Submit Response" to analyze
5. **Continue**: Click "Next Question →" to proceed
6. **Complete**: After all questions, view the comprehensive report

### Supported Audio Formats

- MP3
- MP4
- WAV
- M4A
- WebM
- OGG

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t housing-interview-app .
```

### Run Container
```bash
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_key_here \
  -e LANGSMITH_API_KEY=your_key_here \
  housing-interview-app
```

## ☁️ Cloud Deployment

### Render.com

1. Connect your repository to Render
2. Use the provided `render.yaml` configuration
3. Set environment variables in Render dashboard
4. Deploy!

### Heroku

1. Create a new Heroku app
2. Set environment variables:
```bash
heroku config:set GROQ_API_KEY=your_key_here
heroku config:set LANGSMITH_API_KEY=your_key_here
```
3. Deploy using Git:
```bash
git push heroku main
```

## 🔍 Troubleshooting

### Transcription Not Working

**Issue**: "No transcription method available"

**Solution**: Install at least one transcription library:
```bash
pip install faster-whisper  # Recommended
```

### FFmpeg Not Found

**Issue**: Audio processing fails

**Solution**: Install FFmpeg:
- Windows: Download from ffmpeg.org and add to PATH
- macOS: `brew install ffmpeg`
- Linux: `sudo apt-get install ffmpeg`

### Groq API Errors

**Issue**: "GROQ_API_KEY not found"

**Solution**: 
1. Get API key from [console.groq.com](https://console.groq.com)
2. Add to `.env` file: `GROQ_API_KEY=your_key_here`

### Port Already in Use

**Issue**: "Address already in use"

**Solution**: Change port in `.env` or command line:
```bash
PORT=8002 python main.py
```

## 📊 Project Structure

```
housing-interview-app/
├── main.py                 # FastAPI application
├── templates/
│   └── index.html         # Frontend interface
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── Procfile              # Heroku deployment
├── render.yaml           # Render.com deployment
├── runtime.txt           # Python version
├── .env                  # Environment variables (create this)
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## 🛠️ Development

### Running in Development Mode

```bash
uvicorn main:app --reload --port 8001
```

### Testing API Endpoints

Use the `/health` endpoint to verify setup:
```bash
curl http://localhost:8001/health
```

### Logging

The application uses Python's logging module. Set log level:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 🔐 Security Considerations

- Never commit `.env` file to version control
- Use environment variables for all sensitive data
- Implement authentication for production use
- Validate and sanitize all user inputs
- Use HTTPS in production
- Regularly update dependencies

## 📈 Performance Tips

1. **Use Faster-Whisper**: Best balance of speed and accuracy
2. **Adjust Model Size**: Use "base" for speed, "large" for accuracy
3. **Enable Caching**: Cache transcription results
4. **Use CDN**: Serve static files via CDN in production
5. **Scale Horizontally**: Run multiple instances behind load balancer

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **FastAPI**: Modern web framework
- **Groq**: Fast LLM inference
- **LangChain**: LLM application framework
- **Faster-Whisper**: Efficient speech recognition
- **OpenAI**: Whisper model

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review API documentation

## 🔄 Version History

### v1.0.0 (Current)
- Initial release
- Multi-engine transcription support
- AI-powered analysis with Groq
- Comprehensive reporting
- Responsive web interface
- Docker and cloud deployment support

---

**Built with ❤️ for better housing services**
