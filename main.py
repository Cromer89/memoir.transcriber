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

        transcript = openai.Audio.transcribe("whisper-1", audio_file_obj)
        return jsonify({"transcript": transcript["text"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data or "transcript" not in data:
        return jsonify({"error": "Missing transcript data"}), 400

    try:
        prompt = (
            "Extract key metadata from the following transcript. "
            "Return JSON with 4 fields: 'people', 'places', 'time', 'topics'. "
            "Each field should be a list of keywords.\n\n"
            f"Transcript:\n{data['transcript']}"
        )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a metadata extraction assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        reply = response["choices"][0]["message"]["content"]
        return jsonify({"metadata": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
