from sklearn.linear_model import LinearRegression
import numpy as np

# Sample training data
area = np.array([1000, 1200, 1500, 1800, 2000, 2200]).reshape(-1,1)
price = np.array([200000, 250000, 300000, 350000, 400000, 450000])

# Train model
model = LinearRegression()
model
