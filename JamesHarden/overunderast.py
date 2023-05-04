import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

filepath = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamesHarden\\JamesHardenGameEntireCareer.csv'
data = pd.read_csv(filepath)
data['GAME_DATE'] = pd.to_datetime(data['GAME_DATE'])

data['HOME'] = data['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
data['OPPONENT'] = data['MATCHUP'].apply(lambda x: x.split()[-1])

features = ['HOME', 'REB', 'STL', 'PTS', 'TOV', 'FG3M', 'FG3A', 'BLK']
target = 'AST'

X = data[features]
y = data[target]

# Binarize the target variable based on the threshold of 8.0 assists
y = y.apply(lambda x: 1 if x > 8.0 else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
logistic_regression_model = LogisticRegression()

# Train the model
logistic_regression_model.fit(X_train, y_train)

# Make predictions
predictions = logistic_regression_model.predict(X_test)

# Calculate the evaluation metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f'Logistic Regression - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}\n')

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Will James Harden get Over 8.0 assists?')
plt.show()

# Input the feature values for today's game
# Replace these values with the relevant data for the game
today_game = {
    'HOME': 0,  # 1 for home, 0 for away
    'REB': 4.6,   # Average assists
    'STL': 1.6,   # Average steals
    'PTS': 22.8,   # Average rebounds
    'TOV': 3.4,   # Average turnovers
    'FG3M': 4.2,  # Average made 3-point field goals
    'FG3A': 9.4,  # Average attempted 3-point field goals
    'BLK': 0.2    # Average blocks
}

# Convert the dictionary to a DataFrame
today_game_df = pd.DataFrame([today_game])

# Make the prediction using the trained logistic regression model
today_game_prediction = logistic_regression_model.predict(today_game_df)

# Interpret the prediction
if today_game_prediction[0] == 1:
    print("The model predicts that James Harden will score over 8.0 assists in today's game.")
else:
    print("The model predicts that James Harden will score under 8.0 assists in today's game.")