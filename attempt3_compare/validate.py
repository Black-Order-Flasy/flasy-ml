import pandas as pd
import pickle

# Validation script function
def predict_flood_probability(rr, forest_ratio, streamflow):
    with open('model/flood_probability_best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    features = pd.DataFrame({
        'RR': [rr],
        'Forest_Ratio': [forest_ratio],
        'Streamflow': [streamflow]
    })
    prediction = model.predict(features)
    return prediction[0]

rainfall = float(input("Rainfall: "))
forest_ratio = float(input("Forest Ratio: "))
streamflow = float(input("Streamflow: "))

example_prediction = predict_flood_probability(rainfall, forest_ratio, streamflow)
print(f"Example prediction (flood probability percentage): {example_prediction}")

if (example_prediction >= 1.00 or example_prediction <= 25.00) :
    print("Aman")
elif (example_prediction >= 25.10 or example_prediction <= 50.00) :
    print("Siaga")
elif (example_prediction >= 50.10 or example_prediction <= 75.00) :
    print("Waspada")
elif (example_prediction >= 75.10 or example_prediction <= 100.00) :
    print("Awas")