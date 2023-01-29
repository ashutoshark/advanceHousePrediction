import json
import math 
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
 
app=Flask(__name__)

regmodle=pickle.load(open('regmodle.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=(np.array(list(data.values())).reshape(1,-1))
    output=regmodle.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=int(regmodle.predict(final_input)[0])
    if(output<=1):
       return render_template("home.html",prediction_text=" the flower Iris-Setosa  ".format(output))
    elif(output>=1&output<=2):
       return render_template("home.html",prediction_text=" the flower Iris-Versicolour  ".format(output))
    else:
        return render_template("home.html",prediction_text=" Iris-Virginica ".format(output))


if __name__=="__main__":
    app.run(debug=True)
   
    
   
     

