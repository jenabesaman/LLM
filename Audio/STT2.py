import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from pydub import AudioSegment
from pydub.utils import which
import os

# Set the ffmpeg path if it's not globally available
AudioSegment.converter = which("ffmpeg")  # Ensure ffmpeg is found


# Load Wav2Vec2 model for speech-to-text with automatic progress bar during model download
def load_wav2vec2_model():
    """Download and load the Wav2Vec2 Persian model with automatic progress bar."""
    stt_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-persian"  # Fine-tuned for Persian
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading the Wav2Vec2 model...")
    stt_model = Wav2Vec2ForCTC.from_pretrained(stt_model_name).to(device)
    stt_tokenizer = Wav2Vec2Tokenizer.from_pretrained(stt_model_name)

    return stt_model, stt_tokenizer, device


def convert_to_wav(input_audio_path: str) -> str:
    """Convert .ogg to .wav using pydub and return the new file path."""
    output_wav_path = input_audio_path.replace('.ogg', '.wav')
    audio = AudioSegment.from_ogg(input_audio_path)  # Convert .ogg to .wav
    audio.export(output_wav_path, format="wav")  # Save as .wav file
    return output_wav_path


def test_wav_loading(file_path: str):
    """Test if the .wav file can be loaded using both pydub and torchaudio."""
    try:
        # Test with pydub
        print(f"Trying to load {file_path} with pydub...")
        audio_pydub = AudioSegment.from_wav(file_path)
        print("pydub loaded the file successfully.")

        # Test with torchaudio
        print(f"Trying to load {file_path} with torchaudio...")
        audio_torchaudio, sample_rate = torchaudio.load(file_path)
        print(f"torchaudio loaded the file successfully, sample rate: {sample_rate}.")

    except Exception as e:
        print(f"Error loading audio file: {str(e)}")
        return False
    return True


def transcribe_audio_from_path(file_path: str, output_txt_path: str):
    """Transcribe Persian audio from a file path and save the text to a file."""
    try:
        # Convert .ogg to .wav if necessary
        if file_path.endswith(".ogg"):
            file_path = convert_to_wav(file_path)

        # Test if the .wav file can be loaded
        if not test_wav_loading(file_path):
            raise Exception("Failed to load the .wav file.")

        stt_model, stt_tokenizer, device = load_wav2vec2_model()  # Load model and tokenizer

        # Load the audio file with torchaudio
        audio_input, sample_rate = torchaudio.load(file_path)

        # Resample if necessary
        if sample_rate != 16000:
            transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_input = transform(audio_input)

        # Get the transcription
        input_values = stt_tokenizer(audio_input.squeeze().numpy(), return_tensors="pt",
                                     sampling_rate=16000).input_values
        input_values = input_values.to(device)
        with torch.no_grad():
            logits = stt_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = stt_tokenizer.decode(predicted_ids[0])

        # Save transcription to a text file
        with open(output_txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(transcription)

        print(f"Transcription saved to {output_txt_path}")
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")


# Example usage:
audio_file_path = "./Audio/1.ogg"  # Your .ogg file
output_file_path = "./Transcription/1.txt"  # Where you want to save the transcription
transcribe_audio_from_path(audio_file_path, output_file_path)
