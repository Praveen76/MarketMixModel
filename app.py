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

print('Done1')

model = pickle.load(open('./model.sav','rb'))

fitted_lambda=0.09785996954548055

meanDict={'Internet_LCount': 68.61194029850746,
 'TV_LCount': 45.13636363636363,
 'Radio_LCount': 38.803030303030305,
 'Other_LCount': 41.19402985074627,
 'Internet_Exp': 170618.64104477613,
 'TV_Exp': 158890.0026865672,
 'Radio_Exp': 101947.55268656714,
 'Others_Exp': 23182.348208955224,
 'FL_Amt': 33353612.334995154,
 'Range': 33.0}

stdDict={'Internet_LCount': 65.47957239264514,
 'TV_LCount': 14.411576116816608,
 'Radio_LCount': 12.097106712650906,
 'Other_LCount': 27.820440929543995,
 'Internet_Exp': 122307.46732575353,
 'TV_Exp': 73365.93650074594,
 'Radio_Exp': 41057.893726632305,
 'Others_Exp': 9281.540160142722,
 'FL_Amt': 26433128.11441197,
 'Range': 19.485036994233976}

def Transformation(demoDf):
    for col in demoDf:
        for i, row_value in demoDf[col].iteritems():
            mean=meanDict[col]
            std=stdDict[col]
            newVal=(row_value-mean)/std
            demoDf[col][i] = newVal
    return demoDf


print('Done2')
    
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
                        'Others_Exp':int_features[7] }])
    test=Transformation(test)
    prediction = model.predict(test)
    predTrans=inv_boxcox(prediction,fitted_lambda)
    print(predTrans)
    predTrans=np.round(predTrans,2)
    return render_template('home.html', prediction_text="Predicted funded loan amount : {}".format((predTrans)))

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