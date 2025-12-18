import pickle
import numpy as np
from flask import Flask, request, render_template

# Load model & scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predictdata', methods=['POST'])
def predict_datapoint():

    # Collect form data (ORDER MATTERS)
    temperature = float(request.form['temperature'])
    RH = float(request.form['RH'])
    Ws = float(request.form['Ws'])
    Rain = float(request.form['Rain'])
    FFMC = float(request.form['FFMC'])
    DMC = float(request.form['DMC'])
    ISI = float(request.form['ISI'])
    Classes = float(request.form['Classes'])
    Region = float(request.form['Region'])

    # Create input array (EXACT training order)
    input_data = np.array([[temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

    # Scale input
    scaled_data = standard_scaler.transform(input_data)

    # Predict
    prediction = ridge_model.predict(scaled_data)[0]

    return render_template(
        'home.html',
        prediction=round(prediction, 2)
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
