import numpy as np
import pandas as pd

#importing csv file
data = pd.read_csv('data.csv')
print(data.head())

#check number of rows and columns
shape = data.shape
print("Rows: " + str(shape[0]) + " Columns: "+str(shape[1]))

#Preprocessing Data
#Changing M to 1 and B to 0
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
data.iloc[:,1]= labelencoder_Y.fit_transform(data.iloc[:,1].values)
print(data.head())

#Looking at corelations between features
print("Corelations Between Features")
print(data.iloc[:,1:12].corr())

#Split the data into independent 'X' and dependent 'Y' variables
X = data.iloc[:, 2:31].values
Y = data.iloc[:, 1].values

# Split the dataset into 75% Training set and 25% Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Sclaing (used to normalize the range of independent variables) data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Create a function within many Machine Learning Models
def models(X_train, Y_train):
    # Using Logistic Regression Algorithm to the Training Set
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)

    # Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X_train, Y_train)

    # print model accuracy on the training data.
    print("Accuracies")
    print('Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
    print('Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
    print('Logistic Regression Test Accuracy:', log.score(X_test, Y_test))
    print('Decision Tree Classifier Test Accuracy:', tree.score(X_test, Y_test))
    return log, tree


model = models(X_train, Y_train)

#Predicting (using DecisionTreeClassifier as it has higher accuracy
Y_pred = model[1].predict(X_test)
pdf = pd.DataFrame({'Actual': Y_test.flatten(), 'Predicted': Y_pred.flatten()})
pdf = pdf
print(pdf)


