import numpy as np
from sklearn.linear_model import LinearRegression

# Sample dataset
X = np.array([[1], [2], [3], [4], [5]])   # Independent variable
y = np.array([2, 4, 6, 8, 10])            # Dependent variable

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Prediction
y_pred = model.predict(X)

print("Predicted values:", y_pred)
