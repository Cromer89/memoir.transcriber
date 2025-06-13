import os
import io
import json
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
    try:
        data = request.get_json()
        transcript = data.get("transcript")

        if not transcript:
            return jsonify({"error": "Missing transcript text"}), 400

        prompt = [
            {
                "role": "system",
                "content": (
                    "Extract the following metadata from this transcript: "
                    "people mentioned, places referenced, time indicators (e.g., dates, seasons, ages, time of day), and key themes. "
                    "Return a compact JSON with keys: people, places, times, themes."
                )
            },
            {
                "role": "user",
                "content": transcript
            }
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4",  # downgrade to gpt-3.5-turbo if needed
            messages=prompt,
            temperature=0.3
        )

        message = response["choices"][0]["message"]["content"].strip()

        # Ensure valid JSON is returned even if GPT wraps it in code block
        json_start = message.find("{")
        json_end = message.rfind("}") + 1
        json_string = message[json_start:json_end]

        return jsonify(json.loads(json_string))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
