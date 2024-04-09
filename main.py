import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


data = pd.read_csv('data/data.csv')
print(data)

y = data['diagnosis'] 
X = data.drop(['diagnosis'], axis=1)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)


lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

plt.scatter(X_test[:, 0], y_test, color='blue', label='Original Data')
plt.scatter(X_test[:, 0], y_pred, color='red', label='Predictions')
plt.xlabel('Feature 1')
plt.ylabel('Target')
t_title = plt.title('Predictions vs Original Data')
plt.savefig(f'images/test-{t_title}.png', format='png')
plt.legend()
plt.legend()


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy: .2f}')
print(classification_report(y_test,y_pred))


report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
plt.figure(figsize=(8, 5))
sns.heatmap(report_df.drop(columns=['support']).astype(float), annot=True, cmap='Blues')
c_title = plt.title('Classification Report Heatmap')
plt.savefig(f'images/test-{c_title}.png', format='png')
plt.show()