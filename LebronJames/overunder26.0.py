import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

filepath = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\LebronJames\\LebronJamesGameEntireCareer.csv'
data = pd.read_csv(filepath)
data['GAME_DATE'] = pd.to_datetime(data['GAME_DATE'])

data['HOME'] = data['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
data['OPPONENT'] = data['MATCHUP'].apply(lambda x: x.split()[-1])

features = ['HOME', 'AST', 'STL', 'REB', 'TOV', 'FG3M', 'FG3A', 'BLK']
target = 'PTS'

X = data[features]
y = data[target]

# Binarize the target variable based on the threshold of 25.5 points
y = y.apply(lambda x: 1 if x > 25.5 else 0)

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
plt.title('Confusion Matrix - Will LeBron James get Over 25.5 points?')
plt.show()

# Input the feature values for today's game
# Replace these values with the relevant data for the game
today_game = {
    'HOME': 0,  # 1 for home, 0 for away
    'AST': 5.4,   # Average assists
    'STL': 0.6,   # Average steals
    'REB': 11.0,   # Average rebounds
    'TOV': 3.2,   # Average turnovers
    'FG3M': 1.0,  # Average made 3-point field goals
    'FG3A': 6.6,  # Average attempted 3-point field goals
    'BLK': 1.4    # Average blocks
}

# Convert the dictionary to a DataFrame
today_game_df = pd.DataFrame([today_game])

# Make the prediction using the trained logistic regression model
today_game_prediction = logistic_regression_model.predict(today_game_df)

# Interpret the prediction
if today_game_prediction[0] == 1:
    print("The model predicts that LeBron James will score over 25.5 points in today's game.")
else:
    print("The model predicts that LeBron James will score under 25.5 points in today's game.")