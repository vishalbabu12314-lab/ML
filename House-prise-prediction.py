import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'Area': [1000, 1200, 1500, 1800, 2000, 2200],
    'Bedrooms': [2, 2, 3, 3, 4, 4],
    'Age': [10, 8, 5, 4, 3, 2],
    'Price': [200000
