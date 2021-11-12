
# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
from train import scaler

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
            Pclass = float(request.form['Pclass'])
            Sex = str(request.form['Sex'])
            Age = float(request.form['Age'])
            SibSp = float(request.form['SibSp'])
            Parch = float(request.form['Parch'])
            Fare = float(request.form['Fare'])

            if Sex=="male":
                male=1
                female=0
            else:
                male=0
                female=1

            # input_df = pd.DataFrame(data=[[Pclass, Sex, Age, SibSp,Parch,Fare]], \
            #                         columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])

            # X_input = dmatrix('rate_marriage + age + yrs_married + children + \
            # religious + educ + occupation + occupation_husb', input_df, return_type="dataframe")

            # print(X_input)

            filename = 'Decision_Tree_opti_model.pickle'
            loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage
            # predictions using the loaded model file
            prediction = loaded_model.predict(scaler.transform([[Pclass, Age, SibSp,Parch,Fare, female, male]]))
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