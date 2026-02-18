from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the models
# These must be in the same folder as app.py on GitHub
model = pickle.load(open('Rainfall.pkl', 'rb'))
scaler = pickle.load(open('scale.pkl', 'rb'))
imputer = pickle.load(open('impter.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get values from the HTML form
    features = [float(x) for x in request.form.values()]
    
    # 2. Convert to DataFrame (This fixes the 'SimpleImputer' 1.7.2 crash)
    # Ensure these names match the columns your model was trained on
    cols = ['Temperature', 'Humidity'] 
    feature_df = pd.DataFrame([features], columns=cols)

    # 3. Process the data using the DataFrame
    imputed_data = imputer.transform(feature_df)
    scaled_data = scaler.transform(imputed_data)
    
    # 4. Make prediction
    prediction = model.predict(scaled_data)

    # 5. Return the correct result page
    if prediction[0] == 1:
        return render_template('chance.html')
    else:
        return render_template('noChance.html')

if __name__ == "__main__":
    app.run(debug=True)
