from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample dataset
X = np.array([[1000], [1200], [1500], [1800], [2000], [2200]])
y = np.array([200000, 250000, 300000, 350000, 400000, 450000])

# Model
model = LinearRegression()

# K-Fold setup
kfold = KFold(n_splits=3, shuffle=True, random_state=1)

# Cross validation
scores = cross_val_score(model, X, y, cv=kfold)

print("Cross Validation Scores:", scores)
print("Average Score:", scores.mean())
