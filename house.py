import pandas as pd

df = pd.read_csv('Housing.csv')

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("price", axis=1) 
y = df_encoded["price"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.tight_layout()
plt.show()

print("Enter the following details to predict the house price:")
area = int(input("Area (in sq.ft): "))
bedrooms = int(input("Number of bedrooms: "))
bathrooms = int(input("Number of bathrooms: "))
stories = int(input("Number of stories: "))
parking = int(input("Number of parking spots: "))

mainroad = input("Is the house on the main road? (yes/no): ").strip().lower()
guestroom = input("Is there a guest room? (yes/no): ").strip().lower()
basement = input("Does it have a basement? (yes/no): ").strip().lower()
hotwaterheating = input("Hot water heating available? (yes/no): ").strip().lower()
airconditioning = input("Air conditioning available? (yes/no): ").strip().lower()
prefarea = input("Is it in a preferred area? (yes/no): ").strip().lower()
furnishing = input("Furnishing status (furnished/semi-furnished/unfurnished): ").strip().lower()

new_input = {
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'parking': parking,
    'mainroad_yes': 1 if mainroad == "yes" else 0,
    'guestroom_yes': 1 if guestroom == "yes" else 0,
    'basement_yes': 1 if basement == "yes" else 0,
    'hotwaterheating_yes': 1 if hotwaterheating == "yes" else 0,
    'airconditioning_yes': 1 if airconditioning == "yes" else 0,
    'prefarea_yes': 1 if prefarea == "yes" else 0,
    'furnishingstatus_semi-furnished': 1 if furnishing == "semi-furnished" else 0,
    'furnishingstatus_unfurnished': 1 if furnishing == "unfurnished" else 0
}

input_df = pd.DataFrame([new_input])
input_df = input_df.reindex(columns=X.columns, fill_value=0)

predicted_price = model.predict(input_df)[0]
print(f"\n🏠 Estimated House Price: ₹{int(predicted_price):,}")
