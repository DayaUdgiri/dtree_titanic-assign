
# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
from patsy import dmatrix

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            rate_marriage = float(request.form['rate_marriage'])
            age = float(request.form['age'])
            yrs_married = float(request.form['yrs_married'])
            children = float(request.form['children'])
            religious = float(request.form['religious'])
            educ = float(request.form['educ'])
            occupation = float(request.form['occupation'])
            occupation_husb = float(request.form['occupation_husb'])

            input_df = pd.DataFrame(data=[[rate_marriage, age, yrs_married,children,religious,educ,occupation,occupation_husb]],
                                    columns=["rate_marriage","age", "yrs_married","children","religious","educ","occupation","occupation_husb"])

            X_input = dmatrix('rate_marriage + age + yrs_married + children + \
            religious + educ + occupation + occupation_husb', input_df, return_type="dataframe")

            print(X_input)

            # X_input = X_input.rename(columns=
            #              {'C(occupation)[T.2.0]': 'occ_2',
            #               'C(occupation)[T.3.0]': 'occ_3',
            #               'C(occupation)[T.4.0]': 'occ_4',
            #               'C(occupation)[T.5.0]': 'occ_5',
            #               'C(occupation)[T.6.0]': 'occ_6',
            #               'C(occupation_husb)[T.2.0]': 'occ_husb_2',
            #               'C(occupation_husb)[T.3.0]': 'occ_husb_3',
            #               'C(occupation_husb)[T.4.0]': 'occ_husb_4',
            #               'C(occupation_husb)[T.5.0]': 'occ_husb_5',
            #               'C(occupation_husb)[T.6.0]': 'occ_husb_6'})

            filename = 'Logi_saga_final_model.pickle'
            loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage
            # predictions using the loaded model file
            prediction = loaded_model.predict(X_input)
            print('prediction is', prediction)
            # showing the prediction results in a UI
            return render_template('results.html',prediction=round(prediction[0],2))
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')


if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app