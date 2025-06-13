import os
import io
from flask import Flask, request, jsonify
import openai

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.before_request
def log_request_info():
    print(f"📥 Incoming request: {request.method} {request.path}")
    print(f"Headers: {dict(request.headers)}")
    print(f"Content-Type: {request.content_type}")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        print("❌ No file found in request")
        return jsonify({"error": "Missing file in request"}), 400

    audio_file = request.files["file"]
    print(f"✅ Received file: {audio_file.filename}")

    try:
        audio_bytes = audio_file.read()
        print(f"ℹ️ File size: {len(audio_bytes)} bytes")
        audio_file_obj = io.BytesIO(audio_bytes)
        audio_file_obj.name = audio_file.filename

        transcript = openai.Audio.transcribe("whisper-1", file=audio_file_obj)
        print(f"✅ Transcription result: {transcript}")

        if not transcript or "text" not in transcript:
            print("❌ Transcript missing 'text' key")
            return jsonify({"error": "No transcription returned"}), 502

        return jsonify({"transcript": transcript["text"]})

    except Exception as e:
        print(f"🔥 Exception occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

