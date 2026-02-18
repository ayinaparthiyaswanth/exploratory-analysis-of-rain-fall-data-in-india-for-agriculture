from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the brains
model = pickle.load(open('Rainfall.pkl', 'rb'))
scaler = pickle.load(open('scale.pkl', 'rb'))
imputer = pickle.load(open('impter.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from HTML
    features = [float(x) for x in request.form.values()]
    final_features = scaler.transform(imputer.transform([np.array(features)]))
    prediction = model.predict(final_features)
    
    return render_template('chance.html') if prediction[0] == 1 else render_template('noChance.html')

if __name__ == "__main__":
    app.run(debug=True)