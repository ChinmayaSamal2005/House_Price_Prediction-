from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__,template_folder="template")

# Load your model and columns list
model = joblib.load('model.pkl')
columns = joblib.load('columns.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    sqft = float(request.form['sqft_living'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = float(request.form['bathrooms'])
    floors = float(request.form['floors'])
    waterfront = int(request.form['waterfront'])
    condition = int(request.form['condition'])

    # Create DataFrame from user input
    input_dict = {
        'sqft_living': [sqft],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'floors': [floors],
        'waterfront': [waterfront],
        'condition': [condition]
    }
    input_df = pd.DataFrame(input_dict)

    # Create a DataFrame with the correct structure (all columns)
    full_input = pd.DataFrame(columns=columns)

    # Use concat instead of append (pandas 2.0+)
    full_input = pd.concat([full_input, input_df], ignore_index=True).fillna(0)

    # Make prediction
    prediction = model.predict(full_input)[0]

    return f"<h2>Predicted House Price: ₹{prediction:,.2f}</h2>"


if __name__ == "__main__":
    app.run(debug=True)