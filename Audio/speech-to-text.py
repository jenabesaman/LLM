import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import os


# Load Wav2Vec2 model for speech-to-text with automatic progress bar during model download
def load_wav2vec2_model():
    """Download and load the Wav2Vec2 model with automatic progress bar."""
    stt_model_name = "facebook/wav2vec2-large-xlsr-53"  # Pre-trained model that supports Persian
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # This will automatically show a progress bar if the model is not cached locally
    print("Loading the Wav2Vec2 model...")
    stt_model = Wav2Vec2ForCTC.from_pretrained(stt_model_name).to(device)
    stt_tokenizer = Wav2Vec2Tokenizer.from_pretrained(stt_model_name)

    return stt_model, stt_tokenizer, device


def transcribe_audio_from_path(file_path: str, output_txt_path: str):
    """Transcribe Persian audio from a file path and save the text to a file."""
    try:
        stt_model, stt_tokenizer, device = load_wav2vec2_model()  # Load model and tokenizer

        # Load the audio file
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


# Example usage
audio_file_path = "C:/Workarea/DSTV3.Danadrive.QA.Ai/Audio/1.ogg"
output_file_path = ":C/Workarea/DSTV3.Danadrive.QA.Ai/Audio/transcription.txt"
transcribe_audio_from_path(audio_file_path, output_file_path)
