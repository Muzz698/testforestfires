import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# import ridge regressor and standard scaler pickle
# ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

application = Flask(__name__)
app=application
@app.route("/")
def index():
    return render_template("index.html")

    # return "<h1>Flask is working! This is not a blank page.</h1>"


if __name__=="__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

