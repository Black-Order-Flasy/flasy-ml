import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset 1
dataset = pd.read_csv('dataset/dataset_1.csv')

# Prepare features and target variable
X = dataset[['RR', 'Forest_Ratio', 'Streamflow']]
y = dataset['Flood']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Validate the model
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)
conf_matrix_gb = confusion_matrix(y_test, y_pred_rf)

print(f"Accuracy: {accuracy_rf}")
print("Classification Report:")
print(report_rf)
print("Confusion Matrix:")
print(conf_matrix_gb)

# Save the model to a file
import joblib
joblib.dump(rf_model, 'model/flood_prediction_rf_model.pkl')
