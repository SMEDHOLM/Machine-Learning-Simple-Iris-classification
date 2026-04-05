from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# Load the Iris dataset from scikit-learn.
iris = load_iris()

# Create a DataFrame using the feature data and add the target labels.
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Split the dataset into feature matrix X and target vector y.
X = df[iris.feature_names]
y = df['target']

# Hold out 20% of the data for testing and use a fixed random state for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train a K-Nearest Neighbors classifier.
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Evaluate the trained model on the test set.
accuracy = model.score(X_test, y_test)
print("Accuracy:", f"{accuracy*100:.2f}%")

# Predict the class for a new sample.
example = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(example)
print("Predicted class:", iris.target_names[prediction][0])