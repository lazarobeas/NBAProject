import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

filepath = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\NikolaJokic\\DraymondGreenGameEntireCareer.csv'
data = pd.read_csv(filepath)
data['GAME_DATE'] = pd.to_datetime(data['GAME_DATE'])

data['HOME'] = data['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
data['OPPONENT'] = data['MATCHUP'].apply(lambda x: x.split()[-1])

features = ['HOME', 'AST', 'STL', 'REB', 'TOV', 'FG3M', 'FG3A', 'BLK']
target = 'PTS'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor()
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    print(f'{model_name} - Mean Absolute Error: {mae}, Mean Squared Error: {mse}')

# Add a constant term for the intercept
X_train_with_constant = sm.add_constant(X_train)
X_test_with_constant = sm.add_constant(X_test)

# Fit the Linear Regression model using statsmodels
lr_model_sm = sm.OLS(y_train, X_train_with_constant).fit()

# Print the R-squared and adjusted R-squared
print(f"R-squared: {lr_model_sm.rsquared}")
print(f"Adjusted R-squared: {lr_model_sm.rsquared_adj}")

# Print the p-values for each feature
p_values = lr_model_sm.pvalues
print("P-values for each feature:")
for feature, p_value in zip(['const'] + features, p_values):
    print(f"{feature}: {p_value}")

# Print the summary statistics
print(lr_model_sm.summary())

# Train the Linear Regression model and make predictions
lr_model = models['Linear Regression']
lr_model.fit(X_train, y_train)
predictions = lr_model.predict(X_test)

# Create a scatter plot of true values vs. predicted values
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')

# Add a reference line (45-degree line)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')

plt.title('Linear Regression: True vs. Predicted Values')
plt.show()

# Calculate the residuals
residuals = y_test - predictions

# Create a scatter plot of true values vs. residuals
plt.scatter(y_test, residuals)
plt.xlabel('True Values')
plt.ylabel('Residuals')

# Add a reference line at y = 0
plt.axhline(y=0, color='red', linestyle='--')

plt.title('Linear Regression: Residuals Plot')
plt.show()



# best_model = models['Linear Regression']  # Replace 'Random Forest' with the best model's name
# future_games = pd.read_csv('future_games.csv')  # Read the data for future games
# X_future = future_games[features]
# predictions = best_model.predict(X_future)

