import json
import pickle
form flask import Flask,requests,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
 
app=Flask(__name__)

regmodle=pickle.load(open('regmodel.pkl','rb'))
# scalar=
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=requests.json['data']
    print(data)
    print(np.array(list(data.values()))).reshape(1,-1)
    new_data=
