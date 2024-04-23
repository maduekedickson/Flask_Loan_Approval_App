from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pickled model
model = pickle.load(open("models/random_forest_model.pkl", "rb"))

def predict_loan_approval(data):
    # Use the loaded model to make predictions
    prediction = model.predict(data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Receive data from the form
        gender = request.form['gender']
        married = request.form['married']
        dependents = int(request.form['dependents'])
        education = request.form['education']
        self_employed = request.form['self_employed']
        applicant_income = float(request.form['applicant_income'])
        coapplicant_income = float(request.form['coapplicant_income'])
        loan_amount = float(request.form['loan_amount'])
        loan_amount_term = float(request.form['loan_amount_term'])
        credit_history = float(request.form['credit_history'])
        property_area = request.form['property_area']

        # Mapping input values to numerical values
        gender_map = {'Male': 1, 'Female': 0}
        married_map = {'Yes': 1, 'No': 0}
        education_map = {'Graduate': 1, 'Not Graduate': 0}
        self_employed_map = {'Yes': 1, 'No': 0}
        property_area_map = {'Urban': 0, 'Semiurban': 1, 'Rural': 2}

        # Create a DataFrame from the input data
        new_data = pd.DataFrame({
            'Gender': [gender_map[gender]],
            'Married': [married_map[married]],
            'Dependents': [dependents],
            'Education': [education_map[education]],
            'Self_Employed': [self_employed_map[self_employed]],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_amount_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area_map[property_area]]
        })

        # Make predictions on the new dataset
        prediction = predict_loan_approval(new_data)
        if prediction[0] == 1:
            result = "Loan is Approved 👍"
            css_class = "success"
        else:
            result = "Loan is Rejected 👎"
            css_class = "error"
        
        return render_template('index.html', result=result, css_class=css_class)


if __name__ == '__main__':
    app.run(debug=True)
