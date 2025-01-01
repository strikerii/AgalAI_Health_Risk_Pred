
# Health Risk Prediction API

## Overview

The **Health Risk Prediction API** is a Flask-based application designed to predict **Projected Risk Reduction** and **Outcome Health Score** based on user input. It uses trained Random Forest models to provide accurate results.

## Folder Structure

```
.
├── app.py              # Main application file
├── datasets/           # Directory containing datasets (if needed)
├── documentation/      # Directory containing documentation files
├── models/             # Directory containing trained models and encoders
├── model_input.py      # Contains logic for input preprocessing and prediction
├── __pycache__/        # Directory for cached Python files
```

## Setup and Installation

### Prerequisites

1. Python 3.8 or higher installed on your system.
2. Virtual environment (recommended).

### Steps

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**:
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**:
   Navigate to the `documentation` folder and run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   Execute the `app.py` file to start the Flask application:
   ```bash
   python app.py
   ```

## Using the API

Once the application is running, you can send POST requests to the `/predict_health` endpoint.

### Example Request

Use a tool like **Postman** or cURL to send a POST request with a JSON payload. Here's an example of a JSON input:

```json
{
    "Age": 0.174177,
    "Gender": "Male",
    "Ethnicity": "Hispanic",
    "Region": "Asia",
    "BMI": 0.316820,
    "Hypertension": 1,
    "Diabetes": 1,
    "Omega_3_Intake": 1.716040,
    "Vitamin_D_Intake": 1.069303,
    "Protein_Intake": -1.439767,
    "Genetic_Risk_Score": 1.158636,
    "Diet_Type": "Vegetarian",
    "Years_Followed": 1.248997
}
```

### Example Response

```
Predicted Projected Risk Reduction: 24.86
Predicted Outcome Health Score: 82.47
```

## Notes

- Ensure that the trained models and encoders are present in the `models` folder.
- Use standardized input values for numerical fields.
- Provide valid categorical values for features like `Gender`, `Ethnicity`, `Region`, and `Diet_Type`.

## Contact

For issues or queries, reach out to `strikerii@gmail.com`.
