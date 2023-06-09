from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from joblib import load


app = Flask(__name__)
modelaaron = "C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\AaronGordon\\ou12.5ptsmodel.pkl"

# Initialize the Logistic Regression model
logistic_regression_model = load(modelaaron)

@app.route('/train', methods=['POST'])
def train():
    # The client should send the file path in the body of the POST request
    data = request.get_json()
    filepath = data['filepath']

    # Load the data
    data = pd.read_csv(filepath)
    data['GAME_DATE'] = pd.to_datetime(data['GAME_DATE'])

    # Apply transformations
    data['HOME'] = data['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    data['OPPONENT'] = data['MATCHUP'].apply(lambda x: x.split()[-1])

    # Define features and target
    features = ['HOME', 'AST', 'STL', 'REB', 'TOV', 'FG3M', 'FG3A', 'BLK']
    target = 'PTS'

    X = data[features]
    y = data[target]

    # Binarize the target variable based on the threshold of 7.0 points
    y = y.apply(lambda x: 1 if x > 7.0 else 0)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    logistic_regression_model.fit(X_train, y_train)

    # You may also want to save your trained model to disk so that it can be reloaded later
    # joblib.dump(logistic_regression_model, 'logistic_regression_model.joblib')

    return jsonify({'message': 'Model trained successfully'})

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input features from the request's JSON body
    game = request.get_json()

    # Convert the dictionary to a DataFrame
    game_df = pd.DataFrame([game])

    # Make the prediction using the loaded model
    prediction = logistic_regression_model.predict(game_df)

    # Interpret and return the prediction
    prediction_text = "The model predicts that the player will score over 7.0 points in the game." if prediction[
                                                                                                          0] == 1 else "The model predicts that the player will score under 7.0 points in the game."
    return jsonify({'prediction': prediction_text})

if __name__ == "__main__":
    app.run(debug=True)