# AIML Midterm 2026 — Pearson Correlation Assignment

**Student:** Levan Kalatozishvili  
**Course:** AI & Machine Learning in Cybersecurity  
**Task:** Finding Pearson's Correlation Coefficient and Visualization  

---

## 1. Problem Description

The task was to:

1. Extract the data points (blue dots) from the given online graph.  
2. Calculate Pearson's correlation coefficient.  
3. Visualize the data with a scatter plot.  

---

## 2. Extracted Data

The coordinates of the blue points are:

| X     | Y    |
|-------|------|
| -8.9  | -7   |
| -6.7  | -5   |
| -4    | -2   |
| -2    | 0.8  |
| 1     | 1    |
| 2.6   | 3    |
| 4.5   | 4    |
| 6     | 6.5  |
| 8.9   | 8    |

---

## 3. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Extracted data
x = np.array([-8.9, -6.7, -4, -2, 1, 2.6, 4.5, 6, 8.9])
y = np.array([-7, -5, -2, 0.8, 1, 3, 4, 6.5, 8])

# Pearson correlation coefficient
r = np.corrcoef(x, y)[0, 1]
print("Pearson correlation coefficient:", r)

# Visualization
plt.scatter(x, y)
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Scatter plot of extracted data points")
plt.savefig("scatter_plot.png")  # Save the figure for GitHub
plt.show()

![Scatter Plot](Midterm_Exam/scatter_plot.png)  

4. Results
Pearson correlation coefficient:

0.9890267217492069

Interpretation: The correlation is very strong and positive.

This matches the visual pattern of the scatter plot: points nearly align along a straight line.
5. Scatter Plot

The scatter plot produced by the code shows the positive linear relationship between X and Y:

6. Conclusion

The Pearson correlation coefficient was successfully calculated.

The scatter plot confirms a strong positive correlation.

This workflow is fully reproducible with the provided Python code and data.









# Spam Email Detection - Assignment 2

## 1. Dataset

The dataset used in this project is uploaded to this repository:  
[l_kalatozishvili25_32748.csv](l_kalatozishvili25_32748.csv)

The dataset contains features extracted from emails and a label indicating whether the email is spam (`is_spam = 1`) or legitimate (`is_spam = 0`).  

**Features:**
- `words` – total words in the email  
- `links` – number of hyperlinks  
- `capital_words` – number of capitalized words  
- `spam_word_count` – number of spam-related keywords  
- `is_spam` – label (0 = legitimate, 1 = spam)  

**Dataset inspection:**
```python
import pandas as pd

df = pd.read_csv("l_kalatozishvili25_32748.csv")
print(df.head())
print(df['is_spam'].value_counts())

Example output:
   words  links  capital_words  spam_word_count  is_spam
0    118      2              3                0        0
1    338      6             25                4        1
2    935      4             10                8        1
3    208      2              0                0        0
4     79      0              8                6        0

is_spam
0    1250
1    1250
Name: count, dtype: int64

2. Logistic Regression Model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X = df.drop('is_spam', axis=1)
y = df['is_spam']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

Model coefficients and intercept:
Model Coefficients: [[1.82496832 2.21238496 3.28107248 2.28029156]]
Intercept: [1.85545521]

3. Model Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("Accuracy:", acc)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Spam'],
            yticklabels=['Legitimate', 'Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.savefig("confusion_matrix.png")
plt.show()

Results:
Confusion Matrix:
 [[371   9]
 [ 18 352]]
Accuracy: 0.964


4. Email Classification Function
def classify_email(new_email_features):
    import pandas as pd
    df_new = pd.DataFrame([new_email_features])
    df_scaled = scaler.transform(df_new)
    prediction = model.predict(df_scaled)[0]
    return "Spam" if prediction == 1 else "Legitimate"

5. Example Emails

Spam Email Example:
spam_email = {'words': 500, 'links': 10, 'capital_words': 20, 'spam_word_count': 5}
print(classify_email(spam_email))  # Output: Spam

Explanation: High number of links, capital words, and spam keywords → classified as spam.

Legitimate Email Example:
legit_email = {'words': 120, 'links': 1, 'capital_words': 2, 'spam_word_count': 0}
print(classify_email(legit_email))  # Output: Legitimate

Explanation: Low number of links and capital words, no spam keywords → classified as legitimate.

6. Visualizations
## Visualizations


![Class Distribution](Midterm_Exam/class_distribution.png)  
![Feature Importance](Midterm_Exam/feature_importance.png)   
![My Plot](Midterm_Exam/myplot.png)  


6.1 Class Distribution
plt.figure(figsize=(5,5))
sns.countplot(x='is_spam', data=df)
plt.xticks([0,1], ['Legitimate', 'Spam'])
plt.title("Class Distribution")
plt.ylabel("Number of Emails")
plt.savefig("class_distribution.png")
plt.show()
Insight: Dataset contains an equal number of spam and legitimate emails → no class imbalance.

6.2 Feature Importance
coef = model.coef_[0]
features = X.columns
plt.figure(figsize=(8,5))
sns.barplot(x=features, y=coef)
plt.xticks(rotation=45)
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.ylabel("Coefficient Value")
plt.savefig("feature_importance.png")
plt.show()

Insight: capital_words and links are the most important features influencing spam detection.

7. Source Code

Python script for this project:
spam_email_classifier.py

