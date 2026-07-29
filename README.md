Car Price Prediction

A machine learning project that predicts the resale price of used cars using a Random Forest Regressor trained on historical sales data.

Overview

This project estimates the selling price of a used car based on attributes such as showroom price, kilometers driven, fuel type, seller type, transmission, ownership history, and vehicle age. It also reports which features most influence the predicted price.

The script performs the following steps:

Loads and preprocesses the dataset
Trains a Random Forest Regressor
Evaluates model performance using the R² score
Saves and reloads the trained model using pickle
Plots feature importance
Accepts user input to predict the price of a new car
Dataset

The script expects a CSV file named car data.csv in the project root with the following columns:

Column	Description
1. Car_Name: 	Name of the car (dropped before training)
2. Year:	  Year of manufacture
3. Selling_Price: 	Actual selling price (target variable)
4. Present_Price:	Current showroom price, in lakhs
5. Kms_Driven:	Total kilometers driven
6. Fuel_Type:	Petrol, Diesel, or CNG
7. seller_Type:	Dealer or Individual
8. Transmission:	Manual or Automatic
9. Owner:	Number of previous owners

This project is commonly paired with the Car Dekho / Vehicle Dataset available on Kaggle.

Required libraries:

1. pandas
2. NumPy
3. sklearn.model_selection
4. sklearn.ensemble
5. sklearn.metrics
6. pickle
7. Matplotlib

----------------------
Example:

Give car details:
Current showroom price (lakhs): 8.5
Kilometers driven: 35000
Fuel type (petrol/diesel/cng): petrol
Seller type (dealer/individual): dealer
Transmission (manual/automatic): manual
Number of previous owners: 0
Year of manufacture: 2019

Predicted Selling Price: Rs3.72 Lakhs
