from flask import Flask, request, jsonify, render_template
import pandas as pd
import  numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('haf.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features =  [x for x in request.form.values()]
    final_features = np.array(int_features)
    # form_df =  pd.DataFrame(final_features)
    # form = pd.get_dummies(form_df)
    data_unseen =  pd.DataFrame([final_features], columns=['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status'])
    #print(form)
    print(final_features)
    prediction = model.predict(data_unseen)
    output = prediction
    print(prediction[0])
    if output == 1:
        return render_template('high.html', prediction_text='The chance of stroke is high')
    else:
        return render_template('low.html', prediction_text='The chance of stroke is low')

if __name__ == '__main__':
    app.run(debug=True)



















