# ---------------- Step 1: Import Libraries ----------------
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- Step 2: Load Dataset ----------------
# Sample dataset (replace with CSV for larger data)
data = {
    'EmailText': [
        'Congratulations! You have won a lottery. Claim now!',
        'Meeting scheduled at 10 AM tomorrow',
        'Lowest price for new smartphone, buy now',
        'Can we reschedule our appointment?',
        'You have been selected for a prize!',
        'Please find the attached report'
    ],
    'Label': ['Spam', 'Ham', 'Spam', 'Ham', 'Spam', 'Ham']  # 'Ham' = Not Spam
}
df = pd.DataFrame(data)

# ---------------- Step 3: Data Preprocessing ----------------
# Convert labels to numeric
df['Label_num'] = df['Label'].map({'Ham':0, 'Spam':1})

# Split features and labels
X = df['EmailText']
y = df['Label_num']

# Vectorize text data
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# ---------------- Step 4: Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# ---------------- Step 5: Train the Model ----------------
model = MultinomialNB()
model.fit(X_train, y_train)

# ---------------- Step 6: Evaluate the Model ----------------
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%\n")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred, target_names=['Ham','Spam']))

# ---------------- Step 7: Test with New Emails ----------------
new_emails = [
    'Win a free iPhone now',
    'Are we meeting today for lunch?'
]

new_vectorized = vectorizer.transform(new_emails)
predictions = model.predict(new_vectorized)

for email, pred in zip(new_emails, predictions):
    label = 'Spam' if pred == 1 else 'Ham'
    print(f"Email: {email}\nPredicted Label: {label}\n")
