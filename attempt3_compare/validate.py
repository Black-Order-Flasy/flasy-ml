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