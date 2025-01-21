# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the dataset
df = pd.read_csv('dataset.csv')

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Preprocessing: Convert categorical data (e.g., 'GENDER', 'LUNG_CANCER') to numeric
label_encoder_gender = LabelEncoder()
label_encoder_lung_cancer = LabelEncoder()

# Encode the categorical features (e.g., 'GENDER', 'LUNG_CANCER', etc.)
df['GENDER'] = label_encoder_gender.fit_transform(df['GENDER'])  # 'M' = 1, 'F' = 0
df['LUNG_CANCER'] = label_encoder_lung_cancer.fit_transform(df['LUNG_CANCER'])  # 'YES' = 1, 'NO' = 0

# We assume that the target variable is 'LUNG_CANCER' and the rest are features
X = df.drop('LUNG_CANCER', axis=1)  # Features
y = df['LUNG_CANCER']  # Target variable

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)

# Train the decision tree classifier on the training set
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot the decision tree (optional visualization)
plt.figure(figsize=(12,8))
plot_tree(model, filled=True, feature_names=X.columns, class_names=label_encoder_lung_cancer.classes_, rounded=True)
plt.show()


# Sample data to predict (you can manually fill this as per the input format of your dataset):
Input_data = [
    0,  # GENDER: M (0) for Male
    55,  # AGE: 55
    1,  # SMOKING: 1 (Yes)
    2,  # YELLOW_FINGERS: 2 (High severity)
    2,  # ANXIETY: 2 (High anxiety)
    1,  # PEER_PRESSURE: 1 (Moderate pressure)
    1,  # CHRONIC DISEASE: 1 (Yes)
    2,  # FATIGUE: 2 (High fatigue)
    1,  # ALLERGY: 1 (Present allergy)
    2,  # WHEEZING: 2 (Severe wheezing)
    2,  # ALCOHOL CONSUMING: 2 (Alcohol consumption)
    2,  # COUGHING: 2 (Severe coughing)
    2,  # SHORTNESS OF BREATH: 2 (Severe shortness of breath)
    2,  # SWALLOWING DIFFICULTY: 2 (Severe difficulty)
    2   # CHEST PAIN: 2 (Severe chest pain)
]
    


# Make a prediction using the trained decision tree model
predicted_class = model.predict([Input_data])

# Output the prediction
predicted_label = label_encoder_lung_cancer.inverse_transform(predicted_class)  # Convert numeric prediction back to 'YES'/'NO'
print(f"The model predicts: {predicted_label[0]} (Lung Cancer Prediction)")
