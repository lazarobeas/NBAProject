from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)

# Load and preprocess the data
filepath = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\GabeVincent\\GabeVincentGameEntireCareer.csv'
data = pd.read_csv(filepath)
data['GAME_DATE'] = pd.to_datetime(data['GAME_DATE'], format='%b %d, %Y')
data['HOME'] = data['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
data['OPPONENT'] = data['MATCHUP'].apply(lambda x: x.split()[-1])
features = ['HOME', 'AST', 'STL', 'REB', 'TOV', 'FG3M', 'FG3A', 'BLK', 'FGA', 'FGM', 'FTA', 'FTM', 'PLUS_MINUS', ]
target = 'PTS'
X = data[features]
y = data[target]

@app.route('/predict', methods=['POST'])
def predict():
    data_request = request.get_json()
    point_threshold = float(data_request["point_threshold"])
    today_game = data_request["today_game"]
    y_binarized = y.apply(lambda x: 1 if x > point_threshold else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y_binarized, test_size=0.4, random_state=42)
    logistic_regression_model = LogisticRegression(max_iter=1000)
    logistic_regression_model.fit(X_train, y_train)
    today_game_df = pd.DataFrame([today_game])
    today_game_prediction = logistic_regression_model.predict(today_game_df)
    prediction_text = f"Gabe Vincent will score over {point_threshold} points in today's game." if today_game_prediction[0] == 1 else f"Gabe Vincent will score under {point_threshold} points in today's game."
    return jsonify({'prediction_text': prediction_text})

if __name__ == '__main__':
    app.run(debug=True)



