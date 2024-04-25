import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
data = pd.read_csv("dataset.csv")  # Replace "your_dataset.csv" with the path to your dataset file

# Separate features (X) and target variable (y)
X = data.drop(columns=['results','s.no'])  # Features
y = data['results']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess categorical columns using OneHotEncoder
categorical_cols = ['batting_team', 'bowling_team', 'city']
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first'), categorical_cols)
    ],
    remainder='passthrough'
)

# Define the RandomForestClassifier model
rf_model = RandomForestClassifier(random_state=42)

# Create a pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', rf_model)
])

# Train the model
pipe.fit(X_train, y_train)

# Save the trained model
joblib.dump(pipe, 'rf_model_new.pkl')

# Evaluate the model
train_accuracy = pipe.score(X_train, y_train)
test_accuracy = pipe.score(X_test, y_test)

print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)
