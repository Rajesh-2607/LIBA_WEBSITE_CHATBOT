from flask import Flask, request, jsonify
from flask_cors import CORS
from liba_chatbot import answer_query

app = Flask(__name__)

# Allow CORS only for your frontend domains
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "okay"})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get("message")

        if not user_message:
            return jsonify({"reply": "Please provide a valid message."}), 400

        response = answer_query(user_message)
        return jsonify({"reply": response})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"reply": "Something went wrong processing your query."}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True, port=5001)
