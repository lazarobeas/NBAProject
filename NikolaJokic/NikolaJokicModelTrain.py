import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import dump
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42)

logistic_regression_model = LogisticRegression(max_iter=1000)

logistic_regression_model.fit(X_train, y_train)

# Save the trained model and the scaler to a file
dump(logistic_regression_model, 'jokic_model.joblib')
dump(scaler, 'jokic_scaler.joblib')

