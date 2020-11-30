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
from sklearn.preprocessing import PolynomialFeatures

#from model import encode
import pickle
import operator

app = Flask(__name__)

print('Done1')

model = pickle.load(open('./model.sav','rb'))

polynomial_features= PolynomialFeatures(degree=2)
fitted_lambda=-0.3459788735069217

meanDict={'Internet_LCount': 2339.616666666667,
 'Radio_LCount': 325.21666666666664,
 'TV_LCount': 543.85,
 'Other_LCount': 460.26666666666665,
 'Internet_Exp': 206289.14133333325,
 'Radio_Exp': 88113.31699999998,
 'TV_Exp': 144973.2666666667,
 'Other_Exp': 19821.117833333337,
 'FL_Amt': 24341066233.662113}

stdDict={'Internet_LCount': 779.7722634002581,
 'Radio_LCount': 87.88669355456946,
 'TV_LCount': 181.3443045139853,
 'Other_LCount': 450.9599503682763,
 'Internet_Exp': 78335.36024980673,
 'Radio_Exp': 21768.13628693292,
 'TV_Exp': 54230.19755724163,
 'Other_Exp': 10596.293305193014,
 'FL_Amt': 105480068914.12782}

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
                    'Radio_LCount':int_features[1],
                      'TV_LCount':int_features[2],
                       'Other_LCount':int_features[3],
                        'Internet_Exp':int_features[4],
                        'Radio_Exp'  :int_features[5],
                        'TV_Exp':int_features[6],
                        'Other_Exp':int_features[7] }])
    test=Transformation(test)
    xp = polynomial_features.fit_transform(test)

    prediction = model.predict(xp)
    predTrans=inv_boxcox(prediction,fitted_lambda)
    print(predTrans)
    predTrans=np.round(predTrans,2)
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
