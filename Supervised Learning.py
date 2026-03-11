# Supervised Learning Example (Linear Regression)

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Training data (Hours studied vs Marks)
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([35, 40, 50, 55, 65])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Prediction
pred = model.predict([[6]])

print("Predicted Marks for 6 hours study:", pred)

# Visualization
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.title("Supervised Learning - Linear Regression")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()
