# main.py

import wave
import librosa
import noisereduce as nr
from pydub import AudioSegment
import numpy as np
import os
import whisper
from pyannote.audio import Pipeline
from pyannote.core import Segment
import torch
from dotenv import load_dotenv
from TTS.api import Synthesizer

# Load environment variables from .env file
load_dotenv()

class WavFileReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        with wave.open(self.file_path, 'rb') as wav_file:
            params = wav_file.getparams()
            frames = wav_file.readframes(params.nframes)
            return frames, params

class Preprocessor:
    def __init__(self, frames, params, target_sample_rate):
        self.frames = np.frombuffer(frames, dtype=np.int16)  # Ensure frames is a 1D array with correct dtype
        self.params = params
        self.target_sample_rate = target_sample_rate

    def recreate_original_audio(self):
        # Create the processed_data directory if it doesn't exist
        os.makedirs("./processed_data", exist_ok=True)

        # Recreate the original audio from frames
        original_audio = AudioSegment(
            self.frames.tobytes(),
            frame_rate=self.params.framerate,
            sample_width=self.params.sampwidth,
            channels=self.params.nchannels
        )

        # Save the original audio
        original_audio.export("./processed_data/original_audio.wav", format="wav")

        return original_audio

    def preprocess(self):
        # Recreate and save the original audio
        self.recreate_original_audio()

        # Modify sample rate using librosa for better quality
        target_sample_rate = self.target_sample_rate
        orig_sr = self.params.framerate
        # librosa expects float32 in range [-1.0, 1.0]
        frames_float = self.frames.astype(np.float32) / 32768.0
        resampled_float = librosa.resample(frames_float, orig_sr=orig_sr, target_sr=target_sample_rate)
        # Convert back to int16 range and dtype
        resampled_frames = (resampled_float * 32768).astype(np.int16)

        # Save resampled audio
        resampled_audio = AudioSegment(
            resampled_frames.tobytes(),
            frame_rate=target_sample_rate,
            sample_width=self.params.sampwidth,
            channels=self.params.nchannels
        )
        resampled_audio.export("./processed_data/resampled_audio.wav", format="wav")

        # Eliminate noise
        # Convert int16 to float32 in range [-1, 1] for noise reduction
        resampled_float = resampled_frames.astype(np.float32) / 32768.0
        reduced_noise_float = nr.reduce_noise(
            y=resampled_float,
            sr=target_sample_rate,
            prop_decrease=0.6,        # 60 % of the estimated noise subtracted
            stationary=True           # use stationary algorithm
        )
        # Convert back to int16
        reduced_noise_frames = (reduced_noise_float * 32768).astype(np.int16)

        # Save noise-reduced audio
        noise_reduced_audio = AudioSegment(
            reduced_noise_frames.tobytes(),
            frame_rate=target_sample_rate,
            sample_width=self.params.sampwidth,
            channels=self.params.nchannels
        )
        noise_reduced_audio.export("./processed_data/noise_reduced_audio.wav", format="wav")

        # Handle silence
        audio_segment = AudioSegment(
            reduced_noise_frames.tobytes(),
            frame_rate=target_sample_rate,
            sample_width=self.params.sampwidth,
            channels=self.params.nchannels
        )
        chunk_duration_ms = 20  # Chunk duration in milliseconds
        chunk_size = int(self.target_sample_rate * (chunk_duration_ms / 1000.0))

        non_silent_audio = AudioSegment.silent(duration=0)
        for i in range(0, len(audio_segment), chunk_size):
            chunk = audio_segment[i:i + chunk_size]
            if chunk.dBFS > -60:  # Threshold for silence
                non_silent_audio += chunk

        # Export the final processed audio
        output_file_path = "./processed_data/processed_audio.wav"
        non_silent_audio.export(output_file_path, format="wav")

        return non_silent_audio.get_array_of_samples()

class SpeechToTextConverter:
    def __init__(self, preprocessed_data, sample_rate):
        self.preprocessed_data = preprocessed_data
        self.sample_rate = sample_rate
        self.whisper_model = whisper.load_model("medium")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.getenv("PYANNOTE_API_KEY")
        )

    def convert(self):
        # Ensure preprocessed_data is a numpy array and normalize to [-1, 1]
        audio_data = np.array(self.preprocessed_data, dtype=np.float32) / 32768.0

        # Specify Persian as the language for transcription
        language_code = "fa"  # Persian language code

        # Convert speech to text using Whisper with word timestamps and specified language
        result = self.whisper_model.transcribe(audio_data, word_timestamps=True, language=language_code)
        transcript_segments = result['segments']

        # Prepare audio input for PyAnnote
        audio_input = {
            "waveform": torch.from_numpy(audio_data).unsqueeze(0),  # Add batch dimension
            "sample_rate": self.sample_rate
        }

        # Perform speaker diarization using PyAnnote
        diarization = self.diarization_pipeline(audio_input)

        # Integrate text and speaker labels
        conversation = self.assign_speakers(transcript_segments, diarization)

        return "\n".join(conversation)

    def assign_speakers(self, transcript_segments, diarization, min_overlap_sec=0.7):
        annotated_transcript = []
        for segment in transcript_segments:
            start = segment['start']
            end = segment['end']
            speaker_label = "Unknown"
            segment_range = Segment(start, end)

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Calculate intersection segment
                intersection = turn & segment_range  # intersection is a Segment or None
                if intersection and intersection.duration > min_overlap_sec:
                    speaker_label = speaker
                    break

            annotated_transcript.append(f"[{start:.2f}] {speaker_label}: {segment['text']}")
        
        return annotated_transcript

class TextToSpeechConverter:
    def __init__(self, model_path, config_path):
        self.synthesizer = Synthesizer(model_path, config_path)

    def convert_and_save(self, conversation, output_file="./processed_data/speech_outputs/combined_speech.wav"):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        combined_wavs = []
        for line in conversation:
            # Remove speaker label
            text = line.split(': ', 1)[-1]
            # Convert text to speech
            wavs = self.synthesizer.tts(text)
            combined_wavs.append(wavs)
        # Concatenate all wavs
        combined_wav = np.concatenate(combined_wavs)
        # Save the combined audio file
        self.synthesizer.save_wav(combined_wav, output_file)


if __name__ == "__main__":
    # Example usage
    file_reader = WavFileReader("E:/codes_py/speech_project/data/1.wav")
    frames, params = file_reader.read()  # Unpack the two outputs
    preprocessor = Preprocessor(frames, params, target_sample_rate=16000)  # Pass frames and params to Preprocessor
    preprocessed_data = preprocessor.preprocess()
    converter = SpeechToTextConverter(preprocessed_data, preprocessor.target_sample_rate)
    text = converter.convert()
    print(text)

    # Initialize TextToSpeechConverter
    tts_converter = TextToSpeechConverter("./models/best_model_30824.pth", "./models/config.json")
    # Convert text to speech
    tts_converter.convert_and_save(text.split("\n")) 