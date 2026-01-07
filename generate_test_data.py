"""
Generiert die Trainings- und Testdaten für den Unit-Test.
Ausführung: python generate_test_data.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('Advertising.csv')
X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = df['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

train_data = X_train.copy()
train_data['Clicked on Ad'] = y_train
train_data.to_csv('train_data.csv', index=False)

test_data = X_test.copy()
test_data['Clicked on Ad'] = y_test
test_data.to_csv('test_data.csv', index=False)

print(f"Trainingsdaten gespeichert: {len(train_data)} Zeilen -> train_data.csv")
print(f"Testdaten gespeichert: {len(test_data)} Zeilen -> test_data.csv")
