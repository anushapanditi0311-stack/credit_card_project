from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('models/card_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Build features list in the exact order the model expects:
        # CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY, CNT_CHILDREN, AMT_INCOME_TOTAL, 
        # NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, 
        # OCCUPATION_TYPE, CNT_FAM_MEMBERS, AGE, YEARS_EMPLOYED, UNEMPLOYED
        features = np.array([[
            int(data['gender']),
            int(data['own_car']),
            int(data['own_realty']),
            int(data['children']),
            float(data['income']),
            int(data['income_type']),
            int(data['education']),
            int(data['family_status']),
            int(data['housing_type']),
            int(data['occupation']),
            int(data['family_size']),
            int(data['age']),
            float(data['years_employed']),
            int(data['unemployed'])
        ]])
        
        # Predict target (0 = Safe/Approved, 1 = Risky/Rejected)
        prediction = model.predict(features)[0]
        
        # Invert label: Approved is true if prediction is 0
        approved = int(prediction) == 0
        
        return jsonify({'approved': approved})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)
