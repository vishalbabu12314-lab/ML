import numpy as np
from sklearn.linear_model import LinearRegression

# Hours studied (input)
X = np.array([[1], [2], [3], [4], [5], [6]])

# Marks obtained (output)
y = np.array([35, 40, 50, 55, 65, 70])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict marks for 7 hours study
hours = np.array([[7]])
predicted_marks = model.predict(hours)

print("Predicted Marks:", predicted_marks)
