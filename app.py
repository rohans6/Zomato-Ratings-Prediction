import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)
location_encoder=pickle.load(open("Location_Encoder.pkl",'rb'))
rest_type_encoder=pickle.load(open("Rest_type_Encoder.pkl",'rb'))
model = pickle.load(open('Model.pkl', 'rb'))
@app.route('/',methods=["GET","POST"])
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    order,table,rest_type,location,votes,cost=[x for x in request.form.values()]
    r=location_encoder.transform([location])[0]
    l=rest_type_encoder.transform([rest_type])[0]
    data=np.array([order,table,votes,l,r,cost])
    data=np.reshape(data,(1,6))
    prediction = model.predict(data)

    output = round(prediction[0], 1)

    return render_template('index.html', prediction_text='Your Rating is: {}'.format(output))

if __name__ == "__main__":
    




    app.run(debug=True)