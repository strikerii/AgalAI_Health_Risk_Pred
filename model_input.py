import pandas as pd
import pickle  # For loading encoders, scalers, and models
import json

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

        # Load the saved LabelEncoders and StandardScaler
        with open('models\\label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)

        with open('models\\scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # Encode categorical columns using the saved LabelEncoders
        for column in ['Gender', 'Ethnicity', 'Region', 'Diet_Type']:
            if column in input_data:
                if column in label_encoders:
                    input_data[column] = label_encoders[column].transform(input_data[column])
                else:
                    raise ValueError(f"Unknown category in '{column}'. Please provide a valid value.")

        # Standardize the numerical features using the saved scaler
        numerical_features = ['Age', 'BMI', 'Omega_3_Intake', 'Vitamin_D_Intake',
                              'Protein_Intake', 'Genetic_Risk_Score', 'Years_Followed']
        input_data[numerical_features] = scaler.transform(input_data[numerical_features])

        # Load the saved models
        with open('models\\best_risk_model.pkl', 'rb') as f:
            best_risk_model = pickle.load(f)

        with open('models\\best_health_model.pkl', 'rb') as f:
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
