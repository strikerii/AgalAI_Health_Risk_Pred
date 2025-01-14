import pandas as pd
import pickle  # For loading encoders, scalers, and models
import json
import os  # Added for cross-platform path handling

def predict_health_risk(json_input):
    try:
        # Parse the JSON string into a dictionary
        input_dict = json.loads(json_input)

        # Ensure the JSON input has the required fields
        required_columns = ['Age', 'Gender', 'Ethnicity', 'Region', 'BMI', 'Hypertension',
                            'Diabetes', 'Omega_3_Intake', 'Vitamin_D_Intake', 'Protein_Intake',
                            'Genetic_Risk_Score', 'Diet_Type', 'Years_Followed']

        if not all(col in input_dict for col in required_columns):
            raise ValueError(f"Input JSON is missing one or more required fields: {required_columns}")

        # Convert the dictionary into a DataFrame
        input_data = pd.DataFrame([input_dict])

        # Use cross-platform path handling for models directory
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current script location
        label_encoders_path = os.path.join(base_dir, 'models', 'label_encoders.pkl')
        scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
        risk_model_path = os.path.join(base_dir, 'models', 'best_risk_model.pkl')
        health_model_path = os.path.join(base_dir, 'models', 'best_health_model.pkl')

        # Load the saved LabelEncoders and StandardScaler
        with open(label_encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        # Encode categorical columns using the saved LabelEncoders
        for column in ['Gender', 'Ethnicity', 'Region', 'Diet_Type']:
            if column in input_data:
                if column in label_encoders:
                    input_data[column] = input_data[column].map(
                        lambda x: label_encoders[column].transform([x])[0]
                        if x in label_encoders[column].classes_ else -1
                    )
                    if -1 in input_data[column].values:
                        raise ValueError(f"Unknown value provided for '{column}'. Please check the input.")
                else:
                    raise ValueError(f"Missing encoder for column '{column}'.")

        # Standardize the numerical features using the saved scaler
        numerical_features = ['Age', 'BMI', 'Omega_3_Intake', 'Vitamin_D_Intake',
                              'Protein_Intake', 'Genetic_Risk_Score', 'Years_Followed']
        input_data[numerical_features] = scaler.transform(input_data[numerical_features])

        # Load the saved models
        with open(risk_model_path, 'rb') as f:
            best_risk_model = pickle.load(f)

        with open(health_model_path, 'rb') as f:
            best_health_model = pickle.load(f)

        # Predict the Projected Risk Reduction
        risk_prediction = best_risk_model.predict(input_data)[0]

        # Predict the Outcome Health Score
        health_score_prediction = best_health_model.predict(input_data)[0]

        # Return predictions as a dictionary
        return {
            "Projected_Risk_Reduction": round(risk_prediction, 2),
            "Outcome_Health_Score": round(health_score_prediction, 2)
        }

    except json.JSONDecodeError:
        raise ValueError("Invalid JSON input. Please provide a valid JSON string.")

    except ValueError as e:
        raise ValueError(f"Error in processing input data: {e}")

    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")
