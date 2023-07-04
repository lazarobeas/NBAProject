from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import load

app = Flask(__name__)
CORS(app)

# Load the trained model and the scaler from a file
logistic_regression_model = load('jokic_model.joblib')
scaler = load('jokic_scaler.joblib')


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    line = data['LINE']

    game_features = {
        'HOME': data['HOME'],
        'AST': data['AST'],
        'STL': data['STL'],
        'REB': data['REB'],
        'TOV': data['TOV'],
        'FG3M': data['FG3M'],
        'FG3A': data['FG3A'],
        'BLK': data['BLK'],
        'FGA': data['FGA'],
        'FGM': data['FGM'],
        'FTA': data['FTA'],
        'FTM': data['FTM'],
        'PLUS_MINUS': data['PLUS_MINUS']
    }

    game_df = pd.DataFrame([game_features])
    game_scaled = scaler.transform(game_df)
    predicted_points = logistic_regression_model.predict(game_scaled)

    if predicted_points[0] > line:
        prediction = f"The model predicts that Nikola Jokic will score over {line} points in simulated game."
    else:
        prediction = f"The model predicts that Nikola Jokic will score under {line} points in simulated game."

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)