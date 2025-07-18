# Speech Processing Project

## Overview
This project is designed to process audio files, perform noise reduction, resample audio, and convert speech to text using OpenAI's Whisper model. Additionally, it performs speaker diarization using PyAnnote to identify and label different speakers in the audio.

## Features
- **Noise Reduction**: Reduces background noise from audio files.
- **Resampling**: Changes the sample rate of audio for better quality.
- **Speech-to-Text Conversion**: Converts audio to text using Whisper, with support for multiple languages including Persian.
- **Speaker Diarization**: Identifies and labels different speakers in the audio using PyAnnote.

## Setup

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd speech_project
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your `.wav` audio files in the `data/` directory.
2. Run the main script:
   ```bash
   python app/main.py
   ```
3. The processed audio and text output will be saved in the `processed_data/` directory.

## Configuration
- **Language**: The language for transcription can be set in the `convert` method of the `SpeechToTextConverter` class. For Persian, use the language code `"fa"`.
- **Sample Rate**: The target sample rate can be adjusted in the `Preprocessor` class.

## Dependencies
- `librosa`: For audio processing and resampling.
- `noisereduce`: For noise reduction.
- `pydub`: For audio manipulation.
- `whisper`: For speech-to-text conversion.
- `pyannote.audio`: For speaker diarization.
- `torch`: Required by PyAnnote and Whisper.

## License
This project is licensed under the MIT License.

## Acknowledgments
- OpenAI for the Whisper model.
- PyAnnote for speaker diarization tools.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes. 