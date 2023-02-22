import sklearn
from sklearn.linear_model import LinearRegression
import numpy as np

# some dummy scikit-learn code 
# Generate some random data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create a LinearRegression object
model = LinearRegression()

# Train the model on the data
model.fit(X, y)

# Make a prediction for a new data point
x_new = np.array([[6]])
y_pred = model.predict(x_new)

# Print the prediction
print(f"Prediction: {y_pred[0]}")