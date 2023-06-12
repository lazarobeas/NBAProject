import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

filepath = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\NikolaJokic\\NikolaJokicGameEntireCareer.csv'
data = pd.read_csv(filepath)
data['GAME_DATE'] = pd.to_datetime(data['GAME_DATE'], format='%b %d, %Y')

print("Loading Prediction...\n")

data['HOME'] = data['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
data['OPPONENT'] = data['MATCHUP'].apply(lambda x: x.split()[-1])

features = ['HOME', 'AST', 'STL', 'REB', 'TOV', 'FG3M', 'FG3A', 'BLK', 'FGA', 'FGM', 'FTA', 'FTM', 'PLUS_MINUS']
target = 'PTS'

X = data[features]
y = data[target]

# Ask the user to input their point threshold
point_threshold = float(input("Enter the current PTS line on your favorite platform: "))

# Binarize the target variable based on the user-specified threshold
y = y.apply(lambda x: 1 if x > point_threshold else 0)

# Scale the input features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42)

# Initialize the Logistic Regression model with increased max_iter
logistic_regression_model = LogisticRegression(max_iter=1000)

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
plt.title(f'Confusion Matrix - Nikola Jokic Simulated with {point_threshold} points')
plt.show()

# Input the feature values for today's game
print("Enter Nikola Jokic's stats for the last 5 games:\n")
today_game = {
    'HOME': int(input("Home court advantage? (1 for home, 0 for away): ")),
    'AST': float(input("Average assists: ")),
    'STL': float(input("Average steals: ")),
    'REB': float(input("Average rebounds: ")),
    'TOV': float(input("Average turnovers: ")),
    'FG3M': float(input("Average made 3-point field goals: ")),
    'FG3A': float(input("Average attempted 3-point field goals: ")),
    'BLK': float(input("Average blocks: ")),
    'FGA': float(input("Average field goal attempts: ")),
    'FGM': float(input("Average field goals made: ")),
    'FTA': float(input("Average free throw attempts: ")),
    'FTM': float(input("Average free throws made: ")),
    'PLUS_MINUS': float(input("Average +/-: ")),
}

# Convert the dictionary to a DataFrame
today_game_df = pd.DataFrame([today_game])

# Scale the input features for today's game using the same scaler
today_game_scaled = scaler.transform(today_game_df)

# Make the prediction using the trained logistic regression model
today_game_prediction = logistic_regression_model.predict(today_game_scaled)

# Interpret the prediction
if today_game_prediction[0] == 1:
    print(f"\nThe model predicts that Nikola Jokic will score over {point_threshold} points in today's game.")
else:
    print(f"\nThe model predicts that Nikola Jokic will score under {point_threshold} points in today's game.")