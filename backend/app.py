from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add the parent directory to Python path to import simple_beauty_chatbot
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simple_beauty_chatbot import get_beauty_response

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')

    if not user_input:
        return jsonify({"error": "Message is required"}), 400

    try:
        # Get response from the beauty chatbot
        response_object = get_beauty_response(user_input)
        
        # Create the response JSON
        response = {
            "advice": response_object.advice,
            "product_suggestions": response_object.product_suggestions,
            "tips": response_object.tips,
            "detected_language": response_object.detected_language
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
