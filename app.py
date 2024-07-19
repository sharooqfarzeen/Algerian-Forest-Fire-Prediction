from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Import Standard Scaler and Ridge Regression models
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def submit():
    Temperature=float(request.form.get('Temperature'))
    RH = float(request.form.get('RH'))
    Ws = float(request.form.get('Ws'))
    Rain = float(request.form.get('Rain'))
    FFMC = float(request.form.get('FFMC'))
    ISI = float(request.form.get('ISI'))
    BUI = float(request.form.get('BUI'))
    Region = float(request.form.get('Region'))

    new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,ISI,BUI,Region]])
    result=ridge_model.predict(new_data_scaled)

    return render_template('result.html', results=result[0])  

if __name__=="__main__":
    app.run(host="0.0.0.0") 