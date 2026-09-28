import numpy as np
import pickle
import sys
import sklearn.linear_model._base

# Compatibility for models saved with older scikit-learn versions
sys.modules['sklearn.linear_model.base'] = sklearn.linear_model._base

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict employee salary from the web form.
    """

    experience = int(request.form['experience'])
    test_score = int(request.form['test_score'])
    interview_score = int(request.form['interview_score'])

    final_features = np.array(
        [[experience, test_score, interview_score]]
    )

    prediction = model.predict(final_features)

    output = round(float(prediction[0]), 2)

    return render_template(
        'index.html',
        prediction_text=f'Employee Salary should be $ {output}'
    )


@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    Predict employee salary from an API request.
    """

    data = request.get_json()

    experience = int(data['experience'])
    test_score = int(data['test_score'])
    interview_score = int(data['interview_score'])

    features = np.array(
        [[experience, test_score, interview_score]]
    )

    prediction = model.predict(features)

    output = float(prediction[0])

    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)
