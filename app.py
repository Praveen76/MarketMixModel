# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:00:54 2020

@author: PraveenKumar
"""


import numpy as np
import pandas as pd
import datetime as dt
from scipy.special import boxcox, inv_boxcox

from flask import Flask, request, jsonify, render_template, url_for

#from model import encode
import pickle
import operator


app = Flask(__name__)

model = pickle.load(open('./model.sav','rb'))

print('Params :',model.get_params())
print('Feat  Imp :',model.feature_importances_)
fitted_lambda= -0.3290899304145554

def place_value(number): 
    return ("{:,}".format(number)) 
    
@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():    
    int_features = [float(x)for x in request.form.values()]
    print('int_features',int_features)
    test=pd.DataFrame.from_records([{
                    'Internet_LCount':int_features[0],
                    'TV_LCount':int_features[1],
                      'Radio_LCount':int_features[2],
                       'Other_LCount':int_features[3],
                        'Internet_Exp':int_features[4],
                        'TV_Exp'  :int_features[5],
                        'Radio_Exp':int_features[6],
                        'Other_Exp':int_features[7] }])
    prediction = model.predict(test)
    predTrans=inv_boxcox(prediction,fitted_lambda)
    predTrans=place_value(int(predTrans))
    return render_template('home.html', prediction_text="Predicted funded loan amount : $ {}".format((predTrans)))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict_proba([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)
