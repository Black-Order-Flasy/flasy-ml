import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
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

# Melt the appendix_3 dataset to have consistent structure
appendix_3_melted = appendix_3.melt(id_vars=['Date', 'Year', 'Month'], var_name='Region', value_name='Flood_Pred_Percentage')

# Merge dataset_2 and appendix_3_melted
combined_data = pd.merge(appendix_3_melted, dataset_2, how='left', left_on=['Year', 'Month', 'Region'], right_on=['Year', 'Month', 'Area'])
combined_data = combined_data.drop(columns=['Area', 'High_Pred', 'Moderate_Pred', 'Low_Pred', 'Real_flood_event', 'High_Accuracy', 'Moderate_Accuracy', 'Low_Accuracy'])

# Generate date range for dataset_1 subset
subset_size = len(combined_data)
dataset_1_subset = dataset_1.head(subset_size)
date_range = pd.date_range(start='2022-01-01', periods=subset_size, freq='D')
dataset_1_subset['Date'] = date_range

# Merge combined_data with dataset_1_subset
final_combined_data = pd.merge(combined_data, dataset_1_subset, how='left', on='Date')

# Prepare data for training
X = final_combined_data[['RR', 'Forest_Ratio', 'Streamflow']].dropna()
y = final_combined_data['Flood_Pred_Percentage'].loc[X.index].apply(lambda x: 1 if x > 50 else 0)  # Binarize the target variable for classification

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate different models
models = {
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'SVC': SVC(),
    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5)
}

results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[model_name] = {
        'Confusion Matrix': cm,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    
    print(f"{model_name} Confusion Matrix:\n {cm}")
    print(f"{model_name} Accuracy: {accuracy}")
    print(f"{model_name} Precision: {precision}")
    print(f"{model_name} Recall: {recall}")
    print(f"{model_name} F1 Score: {f1}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

# Save the best model based on Accuracy
best_model_name = max(results, key=lambda k: results[k]['Accuracy'])
best_model = models[best_model_name]
with open('model/flood_probability_best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print(f"Best model: {best_model_name} with Accuracy: {results[best_model_name]['Accuracy']}")

# Example usage of the prediction function
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

# Example prediction
example_prediction = predict_flood_probability(0.0, 0.59, 9.97)
print(f"Example prediction (flood probability): {example_prediction}")
