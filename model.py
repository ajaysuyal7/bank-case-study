# Description: This file contains the code to create a model for the given dataset.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def create_model():
    # Logistic Regression model for binary classification
    model = LogisticRegression()
    return model


#load the dataset
dataset_path = "Bankloans.csv"
data = pd.read_csv(dataset_path)

# Split the dataset into features and target
#train_data= data['default'.isna()]
#test_data = data['default'.isna()==1]

train_data = data[data['default'].notna()]
print(train_data)

test_data = data[data['default'].isna()]
print(test_data)

# Split the dataset into features and target
X = train_data.drop(columns=['default'])
y = train_data['default']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

# Create the model
model = create_model()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the model
import joblib
model_path = "model.pkl"
joblib.dump(model, model_path)
print("Model saved to", model_path)
# Description: This file contains the code to create a model for the given dataset.
