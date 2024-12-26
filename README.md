# Ally

A real-time voice and vision assistant built with LiveKit, designed specifically for blind and visually impaired users. The assistant, named Ally, processes both voice commands and video input to provide contextual assistance.

## Features

- Real-time voice interaction with sub-500ms response time
- Vision capabilities for context-aware assistance
- Intelligent frame selection for optimal image processing
- Streaming text-to-speech responses
- Voice activity detection (VAD)
- Interruption handling for natural conversation flow

## Technical Stack

- LiveKit for real-time communication
- OpenAI GPT for image analysis & natural language processing
- Deepgram for Speech-to-Text
- Silero for Voice Activity Detection
- OpenCV for image processing
- Custom frame selection algorithm using Laplacian variance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kevnantony/ally.git
cd ally
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:

Step 1: Create a .txt file and paste the following things there. Kindly add your actual API Keys right after =.

```
LIVEKIT_URL=(Redacted)
LIVEKIT_API_KEY=(Redacted)
LIVEKIT_API_SECRET=(Redacted)
DEEPGRAM_API_KEY=(Redacted)
OPENAI_API_KEY=(Redacted)
```
Then type this in your terminal:

```bash
mv file.txt .env
```

## Architecture

### Core Components

1. **Voice Processing Pipeline**
   - Silero VAD for voice activity detection
   - Deepgram STT for speech-to-text conversion
   - OpenAI GPT-4 for natural language understanding
   - OpenAI TTS for voice response generation

2. **Vision Processing Pipeline**
   - Real-time video frame capture
   - Frame quality assessment using Laplacian variance
   - Corrupted frame detection and handling
   - Best frame selection algorithm

3. **Communication Layer**
   - LiveKit Room management
   - Track subscription handling
   - Asynchronous event processing
   - Error handling and recovery

### Key Features Implementation

#### Low-Latency Response System
The system achieves sub-500ms Time to First Token (TTFT) through:
- Optimized model selection (gpt-4o-mini)
- Streaming TTS adaptation
- Asynchronous processing pipeline
- Preloaded VAD/STT modules

#### Reliable Image Capture
Image capture reliability is ensured through:
- Buffer size validation
- Frame dimension verification
- Sharpness assessment using Laplacian variance
- Graceful error handling with fallback options

# Setup

First, create a virtual environment, update pip, and install the required packages:

```
$ python3 -m venv ally_env
$ source ally_env/bin/activate
$ pip install -U pip
$ pip install -r requirements.txt
```

You need to set up the following environment variables.

Then, run the assistant:

```
$ python3 assistant.py download-files
$ python3 assistant.py start
or
$ python3 assistant.py dev
```

Finally, you can load the [hosted playground](https://agents-playground.livekit.io/) and connect it.

The assistant will:
1. Connect to the specified LiveKit room
2. Initialize all AI models and processors
3. Start with a greeting
4. Begin processing voice and video input
5. Respond to user queries with context-aware assistance


## Error Handling

The system implements comprehensive error handling:
- Video track timeout detection
- Frame processing error recovery
- Connection error management
- Graceful degradation for missing capabilities

## Performance Considerations

- Optimized for low-latency responses (<500ms TTFT)
- Efficient frame buffer management
- Asynchronous processing for non-blocking operations
- Memory-efficient image processing pipeline

## Features to add 

- Answer at least 10 image questions in the same call, without degradation in response times, with an equal split of images featuring people and those that do not. (TBD)
- ~The implemented model should be streamed and have Time to first Token of <500ms.~ ✅
- ~Ensure images are captured reliably when necessary and intended.~ ✅
- ~README file should be in a way that lets us run the project with as little modifications as possible.~ ✅


### This project is made by Kevin Antony, drop in an [email](mailto:kevinantony.work@gmail.com) for any queries.


