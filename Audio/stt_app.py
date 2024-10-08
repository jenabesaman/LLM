import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from pydub import AudioSegment
from pydub.utils import which

# Set the ffmpeg path if it's not globally available
AudioSegment.converter = which("ffmpeg")

def process_audio(audio_path):
    # Convert audio to WAV if necessary
    if audio_path.endswith('.ogg'):
        output_wav_path = audio_path.replace('.ogg', '.wav')
        audio = AudioSegment.from_ogg(audio_path)
        audio.export(output_wav_path, format="wav")
    else:
        output_wav_path = audio_path

    # Load the model and tokenizer
    stt_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-persian"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stt_model = Wav2Vec2ForCTC.from_pretrained(stt_model_name).to(device)
    stt_tokenizer = Wav2Vec2Tokenizer.from_pretrained(stt_model_name)

    # Load the audio file with torchaudio
    audio_input, sample_rate = torchaudio.load(output_wav_path)

    # Resample if necessary
    if sample_rate != 16000:
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_input = transform(audio_input)

    # Get the transcription
    input_values = stt_tokenizer(audio_input.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values
    input_values = input_values.to(device)
    with torch.no_grad():
        logits = stt_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = stt_tokenizer.decode(predicted_ids[0])

    # Save the transcription to a text file
    output_txt_path = "./transcription.txt"
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(transcription)

    # Load and return the text from the file
    with open(output_txt_path, "r", encoding="utf-8") as f:
        result_text = f.read()
    return result_text

# # Example usage
# audio_file_path = "./1.ogg"  # Replace with your audio file path
# print(process_audio(audio_file_path))
