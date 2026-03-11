import pandas as pd

# Sample dataset
data = {
    'Area': [1000, 1500, 2000, 1200],
    'City': ['Delhi', 'Mumbai', 'Delhi', 'Chennai'],
    'Price': [200000, 300000, 400000, 250000]
}

df = pd.DataFrame(data)

print("Original Data")
print(df)

# Create dummy variables
dummy = pd.get_dummies(df['City'])

# Combine with original data
df = pd.concat([df, dummy], axis=1)

# Drop original categorical column
df = df.drop('City', axis=1)

print("\nData After Dummy Variables")
print(df)
