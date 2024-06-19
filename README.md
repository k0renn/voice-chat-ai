
[![Python application](https://github.com/bigsk1/voice-chat-ai/actions/workflows/python-app.yml/badge.svg)](https://github.com/bigsk1/voice-chat-ai/actions/workflows/python-app.yml)

# Voice Chat AI

Voice Chat AI is a project that allows you to interact with different AI characters using speech. You can choose between various characters, each with unique personalities and voices. You can run all locally, you can use openai for chat and voice, you can mix between the two.

![Ai-Speech](https://imagedelivery.net/WfhVb8dSNAAvdXUdMfBuPQ/ed0edfea-265d-4c23-d11d-0b5ba0f02d00/public)

### Notes about cpu only mode

The CPU Only version doesn't use the XTTS or Checkpoints, in my testing unless using on a GPU it is to resource intensive. So you will have to use OpenAI API for speech, you can use ollama or openai for chat. 

Want to use ollama and have it analyze the screen make sure to download llava in your ollama install, no need to have llava as model in .env it is hardcoded to use when asking for screen analysis, llama3 even with function calling python package wouldn't work so we use llava just for that.  


## Features

- **Supports both OpenAI and Ollama language models**: Choose the model that best fits your needs.
- **Provides text-to-speech synthesis using OpenAI TTS**: Enjoy natural and expressive voices.
- **No typing needed, just speak**: Hands-free interaction makes conversations smooth and effortless.
- **Analyzes user mood and adjusts AI responses accordingly**: Get personalized responses based on your mood.
- **You can, just by speaking, have the AI analyze your screen and chat about it**: Seamlessly integrate visual context into your conversations.
- **Easy configuration through environment variables**: Customize the application to suit your preferences with minimal effort.


## Installation

### Requirements

- Python 3.10
- Microsoft C++ Build Tools on windows
- Microphone
- A sense of humor

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/bigsk1/voice-chat-ai/tree/cpu-only
   cd voice-chat-ai
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\Activate`
   ```

   or use conda just make it python 3.10

   ```bash
   conda create --name voice-chat-ai python=3.10
   conda activate voice-chat-ai

   
   # For CPU-only installations, use:
   pip install -r cpu_requirements.txt
   ```

3. Install dependencies:



   For CPU-only version: ( not advised and may need some additional work )

   ```bash
   pip install -r cpu_requirements.txt
   ```

Need to have Microsoft C++ Build Tools for TTS
[Microsoft Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)


#### Linux CLI Instructions

You can use the following commands to download and extract the files directly into the project directory:

```sh
# Navigate to the project directory
cd /path/to/your/voice-chat-ai

# Download and extract checkpoints.zip
wget https://github.com/bigsk1/voice-chat-ai/releases/download/models/checkpoints.zip
unzip checkpoints.zip -d .

# Download and extract XTTS-v2.zip
wget https://github.com/bigsk1/voice-chat-ai/releases/download/models/XTTS-v2.zip
unzip XTTS-v2.zip -d .
```

## Configuration

1. Rename the .env.sample to `.env` in the root directory of the project and configure it with the necessary environment variables: - The app is controlled based on the variables you add.

```env
# Conditional API Usage: Depending on the value of MODEL_PROVIDER, that's what will be used when run.
# You can mix and match; use local Ollama with OpenAI speech or use OpenAI model with local XTTS, etc.

# Model Provider: openai or ollama
MODEL_PROVIDER=ollama

# Character to use - Options: samantha, wizard, pirate, valleygirl, newscaster1920s, alien_scientist, cyberpunk, detective
CHARACTER_NAME=wizard

# Text-to-Speech Provider 
TTS_PROVIDER=openai

# OpenAI TTS Voice
# Voice options: alloy, echo, fable, onyx, nova, shimmer
OPENAI_TTS_VOICE=onyx

# Endpoints (set these below and no need to change often)
OPENAI_BASE_URL=https://api.openai.com/v1/chat/completions
OPENAI_TTS_URL=https://api.openai.com/v1/audio/speech
OLLAMA_BASE_URL=http://localhost:11434

# OpenAI API Key for models and speech (replace with your actual API key)
OPENAI_API_KEY=sk-proj-1111111111

# Models to use - llama3 works well for local usage.
# OPTIONAL: For screen analysis, if MODEL_PROVIDER is ollama, llava will be used by default.
# Ensure you have llava downloaded with Ollama. If OpenAI is used, gpt-4o works well.
OPENAI_MODEL=gpt-4o
OLLAMA_MODEL=llama3

# The voice speed for XTTS only (1.0 - 1.5, default is 1.1) - cpu only doesn't work leave as is
XTTS_SPEED=1.2

# NOTES:
# List of trigger phrases to have the model view your desktop (desktop, browser, images, etc.).
# It will describe what it sees, and you can ask questions about it:
# "what's on my screen", "take a screenshot", "show me my screen", "analyze my screen", 
# "what do you see on my screen", "screen capture", "screenshot"
```

## Usage

Run the application:

```bash
python app.py
```

### Commands

- To stop the conversation, say "Quit", "Exit", or "Leave".

## Adding New Characters

1. Create a new folder for the character in the project's characters directory.
2. Add a text file with the character's prompt (e.g., `wizard/wizard.txt`).
3. Add a JSON file with mood prompts (e.g., `wizard/prompts.json`).

## Example Character Configuration

`wizard/wizard.txt`

```
You are a wise and ancient wizard who speaks with a mystical and enchanting tone. You are knowledgeable about many subjects and always eager to share your wisdom.
```

`wizard/prompts.json`

```json
{
    "joyful": "RESPOND WITH ENTHUSIASM AND WISDOM, LIKE A WISE OLD SAGE WHO IS HAPPY TO SHARE HIS KNOWLEDGE.",
    "sad": "RESPOND WITH EMPATHY AND COMFORT, LIKE A WISE OLD SAGE WHO UNDERSTANDS THE PAIN OF OTHERS.",
    "flirty": "RESPOND WITH A TOUCH OF MYSTERY AND CHARM, LIKE A WISE OLD SAGE WHO IS ALSO A BIT OF A ROGUE.",
    "angry": "RESPOND CALMLY AND WISELY, LIKE A WISE OLD SAGE WHO KNOWS THAT ANGER IS A PART OF LIFE.",
    "neutral": "KEEP RESPONSES SHORT AND NATURAL, LIKE A WISE OLD SAGE WHO IS ALWAYS READY TO HELP.",
    "fearful": "RESPOND WITH REASSURANCE, LIKE A WISE OLD SAGE WHO KNOWS THAT FEAR IS ONLY TEMPORARY.",
    "surprised": "RESPOND WITH AMAZEMENT AND CURIOSITY, LIKE A WISE OLD SAGE WHO IS ALWAYS EAGER TO LEARN.",
    "disgusted": "RESPOND WITH UNDERSTANDING AND COMFORT, LIKE A WISE OLD SAGE WHO KNOWS THAT DISGUST IS A PART OF LIFE."
}
```

For XTTS find a .wav voice and add it to the wizard folder and name it as wizard.wav , the voice only needs to be 6 seconds long. Running the app will automaticly find the .wav when it has the characters name and use it. 

## Watch the Demo

CPU Only in windows sandbox fresh install using openai for both chat and speech

[![Watch the video](https://img.youtube.com/vi/d5LbRLhWa5c/maxresdefault.jpg)](https://youtu.be/d5LbRLhWa5c)



## License

This project is licensed under the MIT License.
