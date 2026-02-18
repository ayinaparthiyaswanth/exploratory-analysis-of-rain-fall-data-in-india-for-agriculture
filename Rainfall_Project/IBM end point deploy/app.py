from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load ONLY the models we need for prediction
# We skip the imputer because live form data doesn't have missing values
model = pickle.load(open('Rainfall.pkl', 'rb'))
scaler = pickle.load(open('scale.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get values from the HTML form
    # We turn it into a 2D array immediately for the scaler
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # 2. Scale the data (Skip the imputer!)
    # Direct scaling avoids the 'AttributeError' and '_fill_dtype' crashes
    scaled_data = scaler.transform(final_features)
    
    # 3. Make the prediction
    prediction = model.predict(scaled_data)

    # 4. Return the result
    if prediction[0] == 1:
        return render_template('chance.html')
    else:
        return render_template('noChance.html')

if __name__ == "__main__":
    app.run(debug=True)
