from flask import Flask, render_template, flash, request, session, send_file
from flask import render_template, redirect, url_for, request
# from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

import datetime
import sys

import pickle

import mysql.connector

import numpy as np

app = Flask(__name__)
app.config['DEBUG']
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route("/")
def homepage():
    return render_template('index.html')


@app.route("/AdminLogin")
def AdminLogin():
    return render_template('AdminLogin.html')


@app.route("/NewQueryReg")
def NewQueryReg():
    return render_template('NewQueryReg.html')


@app.route("/UploadDataset")
def UploadDataset():
    return render_template('ViewExcel.html')


@app.route("/AdminHome")
def AdminHome():
    return render_template('AdminHome.html')


@app.route("/Search")
def Search():
    return render_template('Search.html')


@app.route("/adminlogin", methods=['GET', 'POST'])
def adminlogin():
    error = None
    if request.method == 'POST':
        if request.form['uname'] == 'admin' and request.form['password'] == 'admin':

            return render_template('AdminHome.html')

        else:
            return render_template('index.html', error=error)





@app.route("/newquery", methods=['GET', 'POST'])
def newquery():
    if request.method == 'POST':

        t1 = request.form['t1']
        t2 = request.form['t2']
        t3 = request.form['t3']
        t4 = request.form['t4']
        t5 = request.form['t5']
        t6 = request.form['t6']
        t7 = request.form['t7']

        filename2 = "Model/Crime-prediction-rfc-model.pkl"
        classifier2 = pickle.load(open(filename2, 'rb'))

        data = np.array([[t1, t2, t3, t4, t5, t6, t7]])
        my_prediction = classifier2.predict(data)
        print(my_prediction[0])
        section = ''

        if (my_prediction) == 0:
            Predict = 'Murder'
            section = 'Section 307 in The Indian Penal Code'

        elif (my_prediction == 1):
            Predict = 'violence'
            section = "Section 326 of the Indian Penal Code"

        elif (my_prediction == 2):
            Predict = 'ChildAbusing'
            section = "POCSO Act punishment is even stricter"

        elif (my_prediction == 3):
            Predict = 'Offence Against a Person'
            section = "Section 228A in The Indian Penal Code"

        elif (my_prediction == 4):
            Predict = 'Mischief'
            section = "Mischief under Section 425 of IPC covers all those acts that cause any damage or destruction to the property resulting in any wrongful loss or damage"

        elif (my_prediction == 5):
            Predict = 'TheftVehicle'
            section = 'Section 378, Indian Penal Code, 1860'

        elif (my_prediction == 6):
            Predict = 'Accident'
            section = 'Section 80 in The Indian Penal Code'

        return render_template('NewQueryReg.html', Predict=Predict, section=section)


@app.route("/excelpost", methods=['GET', 'POST'])
def uploadassign():
    if request.method == 'POST':

        file = request.files['fileupload']
        file_extension = file.filename.split('.')[1]
        print(file_extension)
        # file.save("static/upload/" + secure_filename(file.filename))

        import pandas as pd
        import matplotlib.pyplot as plt
        df = ''
        if file_extension == 'xlsx':
            df = pd.read_excel(file.read(), engine='openpyxl')
        elif file_extension == 'xls':
            df = pd.read_excel(file.read())
        elif file_extension == 'csv':
            df = pd.read_csv(file)

        import seaborn as sns
        sns.countplot(df['TYPE'], label="Count")
        plt.savefig('static/images/out.jpg')
        iimg = 'static/images/out.jpg'

        print(df)

        # import pandas as pd
        import matplotlib.pyplot as plt

        # read-in data
        # data = pd.read_csv('./test.csv', sep='\t') #adjust sep to your needs

        import seaborn as sns
        sns.countplot(df['TYPE'], label="Count")
        plt.show()

        df.TYPE = df.TYPE.map({'Murder': 0,
                               'violence': 1,
                               'ChildAbusing': 2,
                               'Offence Against a Person': 3,
                               'Mischief': 4,
                               'TheftVehicle': 5,
                               'Accident': 6

                               })

        def clean_dataset(df):
            assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
            df.dropna(inplace=True)
            indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
            return df[indices_to_keep].astype(np.float64)

        df = clean_dataset(df)

        # Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
        df_copy = df.copy(deep=True)
        df_copy[['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE']] = df_copy[
            ['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE']].replace(0, np.NaN)

        # Model Building
        from sklearn.model_selection import train_test_split
        # df.drop(df.columns[np.isnan(df).any()], axis=1)
        X = df.drop(columns='TYPE')
        y = df['TYPE']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import classification_report
        classifier = MLPClassifier()
        classifier = GradientBoostingClassifier()
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        print(classification_report(y_test, y_pred))

        filename = 'Model/Crime-prediction-rfc-model.pkl'

        pickle.dump(classifier, open(filename, 'wb'))

        print("Training process is complete Model File Saved!")

        df = df.head(300)

        return render_template('ViewExcel.html', data=df.to_html(), dataimg=iimg)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
