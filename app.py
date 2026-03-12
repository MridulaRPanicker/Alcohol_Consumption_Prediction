from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and the list of feature columns
model = pickle.load(open('model.pkl', 'rb'))
model_columns = pickle.load(open('columns.pkl', 'rb'))

@app.route('/')
def home():
    # Landing page with infographics (ensure you save an image in static/ folder)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get numerical data from form
    beer = float(request.form['beer_servings'])
    spirit = float(request.form['spirit_servings'])
    wine = float(request.form['wine_servings'])
    continent = request.form['continent']

    # 2. Create a DataFrame for the input
    # This ensures OHE columns are handled correctly
    input_df = pd.DataFrame([[beer, spirit, wine]], 
                            columns=['beer_servings', 'spirit_servings', 'wine_servings'])

    # 3. Handle One-Hot Encoding for Continent
    # Initialize all continent columns as 0
    for col in model_columns:
        if "continent_" in col:
            input_df[col] = 0
    
    # Set the selected continent column to 1 (if not the dropped 'first' category)
    continent_col = f'continent_{continent}'
    if continent_col in model_columns:
        input_df.loc[0, continent_col] = 1

    # 4. Ensure column order matches the training set
    input_df = input_df[model_columns]

    # 5. Prediction
    prediction = model.predict(input_df)[0]

    return render_template('index.html', prediction_text=f'Predicted Total Alcohol: {prediction:.2f} Litres')

if __name__ == "__main__":
    app.run(debug=True)