import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import pickle

# Load datasets
dataset_1 = pd.read_csv('dataset/dataset_1.csv')
dataset_2 = pd.read_excel('dataset/dataset_2.xlsx', header=2)
appendix_3 = pd.read_csv('dataset/dataset_3.csv')

# Clean and restructure dataset_2
dataset_2.columns = ['Year', 'Month', 'Area', 'High_Pred', 'Moderate_Pred', 'Low_Pred', 
                     'Real_flood_event', 'NaN1', 'High_Accuracy', 'Moderate_Accuracy', 'Low_Accuracy']
dataset_2 = dataset_2.drop(columns=['NaN1'])
dataset_2['Year'] = dataset_2['Year'].fillna(method='ffill')
dataset_2['Month'] = dataset_2['Month'].fillna(method='ffill')

# Map month names to numbers
month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
dataset_2['Month'] = dataset_2['Month'].map(month_mapping)

# Process Appendix 3
appendix_3['Date'] = pd.to_datetime(appendix_3['Date'])
appendix_3['Year'] = appendix_3['Date'].dt.year
appendix_3['Month'] = appendix_3['Date'].dt.month
appendix_3_melted = appendix_3.melt(id_vars=['Date', 'Year', 'Month'], var_name='Region', value_name='Flood_Pred_Percentage')

# Merge dataset_2_cleaned and appendix_3_melted
combined_data = pd.merge(appendix_3_melted, dataset_2, how='left', left_on=['Year', 'Month', 'Region'], right_on=['Year', 'Month', 'Area'])
combined_data = combined_data.drop(columns=['Area', 'High_Pred', 'Moderate_Pred', 'Low_Pred', 'Real_flood_event', 'High_Accuracy', 'Moderate_Accuracy', 'Low_Accuracy'])

# Generate date range for dataset_1 subset
subset_size = len(combined_data)
dataset_1_subset = dataset_1.head(subset_size)
date_range = pd.date_range(start='2022-01-01', periods=subset_size, freq='D')
dataset_1_subset['Date'] = date_range

# Merge combined_data with dataset_1_subset
final_combined_data = pd.merge(combined_data, dataset_1_subset, how='left', on='Date')

# Train and evaluate different models
X = final_combined_data[['RR', 'Forest_Ratio', 'Streamflow']].dropna()
y = final_combined_data['Flood_Pred_Percentage'].loc[X.index]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'LinearRegression': LinearRegression(),
    'SVR': SVR(),
    'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=5)
}

results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    results[model_name] = rmse
    print(f"{model_name} RMSE: {rmse}")

# Save the best model
best_model_name = min(results, key=results.get)
best_model = models[best_model_name]
with open('flood_probability_best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print(f"Best model: {best_model_name} with RMSE: {results[best_model_name]}")

# Validation script function
def predict_flood_probability(rr, forest_ratio, streamflow):
    with open('flood_probability_best_model.pkl', 'rb') as file:
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
