# Importing the libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression


# Load training dataset
dataset = pd.read_csv('hiring.csv')


# Handle missing values
dataset['experience'] = dataset['experience'].fillna(0)
dataset['test_score'] = dataset['test_score'].fillna(
    dataset['test_score'].mean()
)


# Select features using column names
X = dataset[['experience', 'test_score', 'interview_score']].copy()


# Converting experience words to integer values
def convert_to_int(word):
    word_dict = {
        'zero': 0,
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10,
        'eleven': 11,
        'twelve': 12,
        0: 0
    }

    return word_dict[word]


# Convert experience values to integers
X['experience'] = X['experience'].apply(convert_to_int)


# Select target variable using column name
y = dataset['salary']


# Create and train Linear Regression model
regressor = LinearRegression()

regressor.fit(X, y)


# Save model to disk
pickle.dump(regressor, open('model.pkl', 'wb'))


# Load model to compare the results
model = pickle.load(open('model.pkl', 'rb'))

print(model.predict([[2, 9, 6]]))