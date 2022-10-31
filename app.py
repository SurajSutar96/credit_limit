
from flask import Flask,render_template,request,jsonify
import numpy as np
import pickle
with open('Credit_limit_model.pkl','rb')as f:
    model=pickle.load(f)

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=["POST"])
def predict():
    features=[float(x) for x in request.form.values()]
    array=[np.array(features)]
    pred=model.predict(array)[0]
    return render_template("home.html",predictions=f"Congratulations you got a credit limit upto:- {pred}")
if __name__=="__main__":
    app.run(port='1111',debug=True)