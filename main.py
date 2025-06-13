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
    try:
        if "file" not in request.files:
            print("❌ No file found in request.files")
            return jsonify({"error": "Missing file in request"}), 400

        audio_file = request.files["file"]
        print(f"✅ Received file: {audio_file.filename}")

        audio_bytes = audio_file.read()
        print(f"ℹ️ File size: {len(audio_bytes)} bytes")

        audio_file_obj = io.BytesIO(audio_bytes)
        audio_file_obj.name = audio_file.filename

        print("⏳ Sending to OpenAI Whisper API...")
        transcript = openai.Audio.transcribe("whisper-1", file=audio_file_obj)

        if not transcript:
            print("❌ No transcript returned from Whisper")
            return jsonify({"error": "No transcription returned"}), 502

        print(f"✅ Transcript: {transcript}")
        return jsonify({"transcript": transcript["text"]})

    except Exception as e:
        print(f"🔥 Exception in /transcribe: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
