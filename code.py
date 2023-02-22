import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the data from the API
response = requests.get("https://api.example.com/football_matches")
data = response.json()

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Drop irrelevant columns
df = df.drop(['match_id', 'date'], axis=1)

# Convert the target variable 'result' into binary values
df['result'] = df['result'].replace(['W', 'D', 'L'], [1, 0, -1])

# One-hot encode the categorical variables 'home_team' and 'away_team'
df = pd.get_dummies(df, columns=['home_team', 'away_team'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('result', axis=1), df['result'], test_size=0.2, random_state=42)

# Create and train the logistic regression model
clf = LogisticRegression(random_state=42, solver='lbfgs', multi_class='multinomial')
clf.fit(X_train, y_train)

# Evaluate the model on the testing set
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Make a prediction on a new example
new_example = pd.DataFrame({'home_team_A': [1], 'home_team_B': [0], 'away_team_A': [0], 'away_team_B': [1], 'home_goals': [2], 'away_goals': [1]})
prediction = clf.predict(new_example)
print(f"Prediction: {prediction}")
