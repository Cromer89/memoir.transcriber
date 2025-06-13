import os
import io
from flask import Flask, request, jsonify
import openai

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "Missing file in request"}), 400

    audio_file = request.files["file"]

    try:
        audio_bytes = audio_file.read()
        audio_file_obj = io.BytesIO(audio_bytes)
        audio_file_obj.name = audio_file.filename

        # Run transcription
        transcript = openai.Audio.transcribe("whisper-1", file=audio_file_obj)

        # If transcript is blank, return controlled error
        if not transcript or "text" not in transcript:
            return jsonify({"error": "No transcription returned"}), 502

        return jsonify({"transcript": transcript["text"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
