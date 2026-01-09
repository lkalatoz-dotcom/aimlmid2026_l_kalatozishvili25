import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("l_kalatozishvili25_32748.csv")

# მოკლე ინსპექცია
print(df.head())
print(df['is_spam'].value_counts())  # შეიცვალა 'class' → 'is_spam'

# Features = ყველა column გარდა 'is_spam'
X = df.drop('is_spam', axis=1)

# Labels = 'is_spam' column
y = df['is_spam']

# Train/Test split 70/30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# მოდელის coefficients
print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Model evaluation
y_pred = model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("Accuracy:", acc)

# Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Spam'],
            yticklabels=['Legitimate', 'Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()

# ფუნქცია ახალი email–ის სემპლის დასაფასებლად
def classify_email(new_email_features):
    """
    new_email_features: dict with keys as feature names
    Example: {'words': 100, 'links': 2, 'capital_words': 5, 'spam_word_count': 1}
    """
    import pandas as pd
    df_new = pd.DataFrame([new_email_features])
    df_scaled = scaler.transform(df_new)
    prediction = model.predict(df_scaled)[0]
    return "Spam" if prediction == 1 else "Legitimate"
# Spam email
spam_email = {'words': 500, 'links': 10, 'capital_words': 20, 'spam_word_count': 5}
print("Spam Email Prediction:", classify_email(spam_email))

# Legitimate email
legit_email = {'words': 120, 'links': 1, 'capital_words': 2, 'spam_word_count': 0}
print("Legitimate Email Prediction:", classify_email(legit_email))

plt.figure(figsize=(5,5))
sns.countplot(x='is_spam', data=df)
plt.xticks([0,1], ['Legitimate', 'Spam'])
plt.title("Class Distribution")
plt.ylabel("Number of Emails")
plt.savefig("class_distribution.png")
plt.show()
coef = model.coef_[0]
features = X.columns
plt.figure(figsize=(8,5))
sns.barplot(x=features, y=coef)
plt.xticks(rotation=45)
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.ylabel("Coefficient Value")
plt.savefig("feature_importance.png")
plt.show()
