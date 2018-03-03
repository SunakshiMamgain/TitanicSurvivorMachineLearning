import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import warnings
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

data = pd.read_csv('C:/Users/Hp/Desktop/New folder/train.csv')

#print(data.info())
"""cabin data is missing. so we can do 3 actions either leave it as it is or take a mean or put a null value to it. we will drop this from thr data and also the survived lable also.
we will make survived data as target value.
we are taking the mean of age and fill the blanked spaces of age field."""
data.Age.fillna(value = np.round(np.mean(data.Age)), inplace = True)
# Embarked
# only in data, fill the two missing values with the most occurred value, which is "S".
data["Embarked"] = data["Embarked"].fillna("S")
input_data = pd.get_dummies(data.drop(['Cabin', 'Survived','Name', 'Ticket', 'PassengerId'], axis = 1))
#print(input_data.head())
#print(input_data.info())
test_data = pd.read_csv('C:/Users/Hp/Desktop/New folder/test.csv')
test_data['Age'].fillna(value = np.mean(test_data['Age']), inplace = True)
test_data['Fare'].fillna(value = np.mean(test_data['Fare']), inplace = True)
input_test_data = pd.get_dummies(test_data.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis = 1))
train_y = data.Survived
#model
RndmFrst= RandomForestClassifier()
RndmFrst.fit(input_data, train_y)
#Predicting
predicted_prices = RndmFrst.predict(input_test_data)
accuracy=RndmFrst.score(input_data, train_y)
print("Accuracy is:", accuracy)
my_submission = pd.DataFrame({'Popularity': predicted_prices})
cross_val=cross_val_score(RndmFrst, input_data, train_y, cv=3, scoring="accuracy")
print("Cross Validation:",cross_val)
y_train_pred = cross_val_predict(RndmFrst, input_data, train_y, cv=3)
conf_mx=confusion_matrix(train_y, y_train_pred)
print("Confussion Matrix:",conf_mx)
ps=precision_score(train_y, y_train_pred,average="macro")
print("precision score:",ps)
rs=recall_score(train_y, y_train_pred,average="macro")
print("recall score:",rs)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
#Cerating seperate data
my_submission.to_csv('submissionSurvivedLogisticRegression.csv', index=False)