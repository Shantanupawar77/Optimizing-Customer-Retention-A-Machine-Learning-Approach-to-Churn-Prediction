from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model and transformer
model = joblib.load('model.pkl')
trans = joblib.load('transformer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    contract = request.form['contract']
    onlinesecurity = request.form['onlinesecurity']
    techsupport = request.form['techsupport']
    internetservice = request.form['internetservice']
    onlinebackup = request.form['onlinebackup']
    tenure = int(request.form['tenure'])
    monthlycharges = float(request.form['monthlycharges'])
    totalcharges = float(request.form['totalcharges'])

    # Create a dictionary from the form data
    cust = {'contract': contract,
            'onlinesecurity': onlinesecurity,
            'techsupport': techsupport,
            'internetservice': internetservice,
            'onlinebackup': onlinebackup,
            'tenure': tenure,
            'monthlycharges': monthlycharges,
            'totalcharges': totalcharges}

    # Create a DataFrame from the dictionary
    cust_df = pd.DataFrame(cust, index=[0])

    # Transform the DataFrame using the transformer
    cust_transformed = trans.transform(cust_df)

    # Predict churn using the model
    prediction = model.predict(cust_transformed)[0]

    # # Determine the result message
    # result_message = 'Churn' if prediction == 1 else 'Not Churn'
    # Determine the result message
    result_message = 'The customer is predicted to churn from the company.' if prediction == 1 else 'The customer is predicted to stay with the company.'
    return render_template('index.html', pred=result_message)

if __name__ == '__main__':
    app.run(debug=True)

