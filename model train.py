import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle


data = pd.read_csv("house_price.csv")
x = data[['size_sqft', 'rooms', 'age', 'location_score']]
y = data['price']

model = LinearRegression()
model.fit(x, y)

# Save model
with open("house_model.pkl", "wb") as f:
    pickle.dump(model, f)