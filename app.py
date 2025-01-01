from flask import Flask, request, jsonify
from flask_cors import CORS
from model_input import predict_health_risk
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict_health', methods=['POST'])
def predict_health():
    try:
        # Parse JSON input from the request
        user_input_json = request.get_json()

        if not user_input_json:
            return "Invalid input. Please provide a valid JSON payload.", 400

        # Pass the JSON input to the prediction function
        result = predict_health_risk(json.dumps(user_input_json))

        # Return the predictions in the old output format
        output = (
            f"Predicted Projected Risk Reduction: {result['Projected_Risk_Reduction']}\n"
            f"Predicted Outcome Health Score: {result['Outcome_Health_Score']}"
        )
        return output, 200

    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
    