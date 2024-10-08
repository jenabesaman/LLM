from flask import Flask, request, jsonify
from stt_app import process_audio  # Import the function from tts_app.py
import os
import base64

app = Flask(__name__)
import subprocess

def convert_ogg_to_wav(input_path, output_path):
    """Converts an OGG file to a WAV file using FFmpeg."""
    command = [
        "ffmpeg", "-y", "-i", input_path, output_path
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg failed: {e}")

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """API endpoint to transcribe Persian audio from a base64 string."""
    try:
        data = request.get_json(force=True)

        # Check if base64_string is provided
        base64_string = data.get("base64_string")
        if not base64_string:
            return jsonify({'error': 'base64_string is required'}), 400

        # Decode the base64 string and save it as an OGG file
        try:
            audio_data = base64.b64decode(base64_string)
            audio_path = "./uploads/audio_input.ogg"
            with open(audio_path, "wb") as f:
                f.write(audio_data)
        except Exception as decode_error:
            return jsonify({'error': f'Base64 decoding failed: {str(decode_error)}'}), 500

        # Convert the OGG file to WAV
        wav_path = "./uploads/audio_input.wav"
        try:
            convert_ogg_to_wav(audio_path, wav_path)
        except Exception as e:
            return jsonify({'error': f'Conversion to WAV failed: {str(e)}'}), 500

        # Call the process_audio function using the WAV file
        transcription = process_audio(wav_path)
        return jsonify({'transcription': transcription})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    os.makedirs("./uploads", exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
