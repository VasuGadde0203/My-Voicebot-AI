# VoiceBot AI
Welcome to VoiceBot AI, an interactive voice-based assistant built with FastAPI, AssemblyAI, and Azure OpenAI. This application allows users to interact via voice input or predefined questions, receiving responses in both text and audio format. It leverages speech-to-text and text-to-speech technologies to create a conversational experience, with support for a resume-based persona.

## Features
- **Voice Interaction:** Record audio input and receive transcribed responses.
- **Text-to-Speech:** Convert responses to audio files with a configurable voice.
- **Resume-Based Responses:** Answers questions based on a provided resume text.
- **Multiple APIs:** Supports AssemblyAI for speech-to-text with fallback to Google Speech Recognition.
- **Real-Time Chat:** Displays conversation in a responsive web interface.
- **Audio Playback:** Play back assistant responses directly in the browser.

## Prerequisites
- **Python** 3.11+
- **Git** (for cloning the repository)
- **Node.js** (optional, for local testing of static files)
- **API Keys:**
  - Azure OpenAI API Key
  - Azure OpenAI Endpoint
  - Azure OpenAI API Version
  - AssemblyAI API Keys (3 keys for redundancy)
- **Google Cloud Text-to-Speech** (optional, if using alternative TTS)
- **Internet Connection** (for API calls)

## Installation
**1. Clone the Repository**
- git clone https://github.com/your-username/voicebot-ai.git
- cd voicebot-ai

**2. Set Up Environment Variables**
- Create a .env file in the project root and add the following variables:
  - AZURE_OPENAI_API_KEY=your_azure_openai_api_key
  - AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
  - AZURE_OPENAI_API_VERSION=your_azure_openai_api_version
  - ASSEMBLY_AI_APIKEY_1=your_assemblyai_apikey_1
  - ASSEMBLY_AI_APIKEY_2=your_assemblyai_apikey_2 (optional, if not remove the loop in the speech to text)
  - ASSEMBLY_AI_APIKEY_3=your_assemblyai_apikey_3  (optional, if not remove the loop in the speech to text)
 
**3. Install Dependencies**
Install the required Python packages:
  - pip install -r requirements.txt
Ensure requirements.txt includes:
  - fastapi
  - uvicorn
  - speechrecognition
  - assemblyai==0.25.0
  - python-multipart
  - gTTS
  - openai
  - pydub
  - python-dotenv

**4. Prepare Resume Text**
- Create a resume.txt file in the project root with your resume content (e.g., education, skills, experiences). This will be used to personalize the bot's responses.

**5. Set Up Static Files**
- Ensure the static directory exists with index.html inside it.
- Create an audio subdirectory under static to store generated audio files:
  - mkdir -p static/audio
 
## Running Locally
**1. Start the FastAPI Server**
Run the application using uvicorn:
  - uvicorn main:app --host 0.0.0.0 --port 8000 --reload
--host 0.0.0.0 allows access from any network interface.
--reload enables auto-reloading on code changes (useful for development).

**2. Access the Web Interface**
- Open your browser and navigate to:
  - You should see the VoiceBot AI interface with a mic button and question cards.
  - Click the mic button to record audio, or click a question card to send a predefined question.
 
**3. Test the API Endpoints**
- **Process Audio:** Upload an audio file to /process_audio (POST) with a file field and optional messages JSON.
- **Ask Question:** Send a POST request to /ask_question with a JSON body:
  - {
  "question": "What’s your #1 superpower?",
  "messages": []
  }
- **Use tools like Postman or curl to test:**
  - curl -X POST "http://localhost:8000/ask_question" -H "Content-Type: application/json" -d '{"question": "What’s your #1 superpower?", "messages": []}'
 
## Usage
- **Recording Audio:** Click the mic button to start/stop recording. The bot will transcribe your speech and respond.
- **Predefined Questions:** Click any question card to ask it and hear the response.
- **Audio Playback:** Click the play icon (▶️) next to assistant messages to listen.

## Project Structure
- **main.py:** FastAPI application logic, including API endpoints and helper functions.
- **static/index.html:** Web interface with JavaScript for real-time interaction.
- **resume.txt:** Resume text for persona-based responses.
- **static/audio/:** Directory for storing generated audio files.

## Deployment
- This project is designed to be deployed on Render. Follow these steps:
  - Push your code to a GitHub repository.
  - Create a new Web Service on Render.
  - Connect your GitHub repository.
  - Set environment variables in Render Dashboard matching the .env file.
  - Deploy and monitor logs for any issues.

## Contributing
- Feel free to submit issues or pull requests. Contributions to improve voice quality, add new features, or optimize performance are welcome!

## Troubleshooting
- **API Errors:** Ensure all API keys are valid and quotas are sufficient.
- **Audio Issues:** Check file permissions and ensure the static/audio directory is writable.
- **CORS Issues:** Verify allow_origins in CORS middleware if testing from a different domain.
- **Logs:** Review main.log or Render logs for detailed error messages.

## Notes
- Replace your-username in the clone URL with your actual GitHub username.
- Adjust the requirements.txt content if you’ve added or removed dependencies (e.g., google-cloud-texttospeech if used).
- The README assumes a basic setup; customize sections like "Contributing" as needed.
