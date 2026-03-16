from flask import Flask, request, jsonify
from flask_cors import CORS
from liba_chatbot import answer_query
from waitress import serve
import config

app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    bot_response = answer_query(user_message)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    print(f"Starting server on port {config.SERVER_PORT}...")
    serve(app, host="0.0.0.0", port=config.SERVER_PORT)
