import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

df = pd.read_csv("car data.csv")

df["Car_Age"] = 2026 - df["Year"]
df = df.drop(["Car_Name", "Year"], axis=1)

df["Fuel_Type"] = df["Fuel_Type"].map(
    {"Petrol": 0, "Diesel": 1, "CNG": 2})
df["Seller_Type"] = df["Seller_Type"].map(
    {"Dealer": 0, "Individual": 1})
df["Transmission"] = df["Transmission"].map(
    {"Manual": 0, "Automatic": 1})

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)
print(f"Accuracy: {r2_score(y_test, model.predict(X_test))*100:.1f}%")


with open("car_price_model.pkl", "wb") as f:   
    pickle.dump(model, f)
print("Model saved! ")


with open("car_price_model.pkl", "rb") as f:   
    model = pickle.load(f)
print("Model loaded! ")


features = X.columns.tolist()
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(features)),
        importance[indices],
        color="steelblue")
plt.xticks(range(len(features)),
           [features[i] for i in indices],
           rotation=45)
plt.title("Which Factors Affect Car Price Most?")
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

print("\n=== Feature Importance Ranking ===")
for i in indices:
    print(f"{features[i]:20} → {importance[i]:.4f}")


def predict_price(present_price, kms_driven, fuel_type,
                  seller_type, transmission, owner, year):
    car_age = 2026 - year

    fuel_map   = {"petrol": 0, "diesel": 1, "cng": 2}
    seller_map = {"dealer": 0, "individual": 1}
    trans_map  = {"manual": 0, "automatic": 1}

    fuel   = fuel_map.get(fuel_type.lower(), 0)
    seller = seller_map.get(seller_type.lower(), 0)
    trans  = trans_map.get(transmission.lower(), 0)

    car_input = pd.DataFrame([[
        present_price, kms_driven, fuel,
        seller, trans, owner, car_age
    ]], columns=["Present_Price", "Kms_Driven", "Fuel_Type",
                 "Seller_Type", "Transmission", "Owner", "Car_Age"])

    predicted = model.predict(car_input)[0]
    return round(predicted, 2)


print("\nGive car details:")
try:
    present_price = float(input("Current showroom price (lakhs): "))
    kms_driven    = int(input("Kilometers driven: "))
    fuel_type     = input("Fuel type (petrol/diesel/cng): ")
    seller_type   = input("Seller type (dealer/individual): ")
    transmission  = input("Transmission (manual/automatic): ")
    owner         = int(input("Number of previous owners: "))
    year          = int(input("Year of manufacture: "))

    result = predict_price(     
        present_price, kms_driven, fuel_type,
        seller_type, transmission, owner, year
    )
    print(f"\nPredicted Selling Price: Rs{result} Lakhs")

except ValueError as e:
    print(f"Invalid input: {e}")
