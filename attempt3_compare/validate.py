import joblib
import numpy as np

# Load the model
rf_model = joblib.load('model/flood_prediction_rf_model.pkl')

# Function to predict flood probability using Random Forest
def predict_flood_probability_rf(rainfall, forest_ratio, streamflow):
    prob = rf_model.predict_proba([[rainfall, forest_ratio, streamflow]])[0][1]
    return prob

# Example usage
rainfall = float(input("Enter Rainfall (RR): "))
forest_ratio = float(input("Enter Forest Ratio: "))
streamflow = float(input("Enter Streamflow: "))

flood_probability = predict_flood_probability_rf(rainfall, forest_ratio, streamflow)
print(f"Flood probability for RR={rainfall}, Forest_Ratio={forest_ratio}, Streamflow={streamflow}: {flood_probability:.2%}")

print(flood_probability)

if (flood_probability >= 0.00 and flood_probability <= 0.25) :
    print("Aman")
elif (flood_probability >= 0.251 and flood_probability <= 0.5) :
    print("Siaga")
elif (flood_probability >= 0.51 and flood_probability <= 0.75) :
    print("Waspada")
elif (flood_probability >= 0.751 and flood_probability <= 1) :
    print("Awas")
