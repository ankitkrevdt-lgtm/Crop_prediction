import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('Crop_recommendation.csv')  # Put CSV in same folder
print(data.shape)
print(data.isnull().sum())

# Split features and labels
x = data.iloc[:, :-1]  # features
y = data.iloc[:, -1]   # labels

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(x_train, y_train)

prediction = model.predict(x_train)
accuracy = model.score(x_test, y_test)
print("Accuracy: ", accuracy)

# FIXED: Correct pickle syntax
pickle.dump(model, open('model.pkl', 'wb'))
print("Model saved as model.pkl")
