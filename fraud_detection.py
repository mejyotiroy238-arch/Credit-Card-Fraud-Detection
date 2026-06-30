import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("dataset/creditcard.csv")

# Display first five rows
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Count of normal and fraudulent transactions
print("\nClass Distribution:")
print(df['Class'].value_counts())

# Visualize class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)

plt.title("Credit Card Fraud Distribution")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.savefig("images/class_distribution.png")
plt.close()

print("\nColumn Names:")
print(df.columns.tolist())

# Remove any extra spaces from column names
df.columns = df.columns.str.strip()

X = df.drop('Class', axis=1)
y = df['Class']

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    solver='liblinear',
    random_state=42,
    max_iter=500
)

model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Predict on test data
y_pred = model.predict(X_test)

print("\nPredictions completed!")

# Model Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("images/confusion_matrix.png")
plt.close()

# Predict a sample transaction
sample = X_test.iloc[0:1]

prediction = model.predict(sample)

if prediction[0] == 0:
    print("\nPrediction: Normal Transaction")
else:
    print("\nPrediction: Fraudulent Transaction")

# Save the trained model
joblib.dump(model, "credit_card_fraud_model.pkl")

print("\nModel saved successfully!")