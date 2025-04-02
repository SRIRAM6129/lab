import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples per class
samples_per_class = 50

# Define class labels
classes = ["Setosa", "Versicolor", "Virginica"]

data = []
for class_label in classes:
    sepal_length = np.random.uniform(4.3, 7.9, samples_per_class)
    sepal_width = np.random.uniform(2.0, 4.4, samples_per_class)
    petal_length = np.random.uniform(1.0, 6.9, samples_per_class)
    petal_width = np.random.uniform(0.1, 2.5, samples_per_class)

    for i in range(samples_per_class):
        data.append([sepal_length[i], sepal_width[i], petal_length[i], petal_width[i], class_label])

# Create DataFrame
iris_df = pd.DataFrame(data, columns=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Class"])

# Print the dataset
print("Generated Iris Dataset:")
print(iris_df.head(10))  # Print first 10 rows for preview

# Encode class labels
le = LabelEncoder()
iris_df["Class"] = le.fit_transform(iris_df["Class"])

# Split dataset into features and target
X = iris_df.drop(columns=["Class"])
y = iris_df["Class"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train k-NN classifier
k = 5  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Identify correct and incorrect predictions
correct_predictions = (y_pred == y_test).sum()
incorrect_predictions = (y_pred != y_test).sum()

# Print evaluation results
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
print(f"Correct Predictions: {correct_predictions}")
print(f"Incorrect Predictions: {incorrect_predictions}")

