#Ruishi Tao
#ITP499 Fall2021
#HW2

#Create a DataFrame “diabetes_knn” to store the diabetes data.

import pandas as pd
import numpy as np
from sklearn.metrics import plot_confusion_matrix
diabetes_knn = pd.read_csv("diabetes.csv")
df=pd.DataFrame(diabetes_knn)
print(df)
#Determine the dimensions of the “diabetes_knn” dataframe.
print(diabetes_knn.shape)
#Update the DataFrame to account for missing values.
print(df.isna())
print(df.isna().sum())
#Create the Feature Matrix (X) and Target Vector (y).
X=df.drop('Outcome',axis=1)
y=df['Outcome']
#Standardize the attributes of Feature Matrix (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
print(scaler.fit(X))
X= pd.DataFrame(scaler.fit_transform(X),columns=X.columns)
print(X)
#Split the Feature Matrix and Target Vector into three partitions. Training A, Training B and test. They should be in the ratio 60-20-20. random_state = 2021, stratify = y
from sklearn.model_selection import train_test_split
X_trainA, X_temp, y_trainA, y_temp = train_test_split(X, y, test_size=0.4, random_state=2021, stratify=y)
X_trainB, X_test, y_trainB, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=2021, stratify=y_temp)
#Develop a KNN based model based on Training A for various ks. K should range between 1 and 30.
from sklearn.neighbors import KNeighborsClassifier
neighbors=np.arange(1,31)
trainA_accuracy=np.empty(30)
trainB_accuracy=np.empty(30)
for k in neighbors:
    print(k)
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_trainA,y_trainA)
    y_pred=knn.predict(X_trainB)
#Compute the KNN score (accuracy) for training A and training B data for those ks.
    trainA_accuracy[k-1]=knn.score(X_trainA,y_trainA)
    trainB_accuracy[k-1]=knn.score(X_trainB,y_trainB)
    print(trainA_accuracy[k-1])
    print(trainB_accuracy[k-1])
#Plot a graph of training A and training B accuracy and determine the best value of k. Label the plot.
import matplotlib.pyplot as plt
from sklearn import metrics
plt.figure(2)
plt.title("KNN:varying number of neighbors")
plt.plot(neighbors,trainB_accuracy,label="Training B Accuracy")
plt.plot(neighbors,trainA_accuracy,label="Training A Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()
#we can therefore select k to be 14
#Now, using the selected value of k, score the test data set
knn=KNeighborsClassifier(n_neighbors=14)
model=knn.fit(X_trainA,y_trainA)
print(model.score(X_test,y_test))
#Plot the confusion matrix (as a figure)

#cf= metrics.confusion_matrix(y_trainB,y_pred)
plot_confusion_matrix(model,X_test,y_test)
plt.show()
#Predict the Outcome for a person with 2 pregnancies, 150 glucose, 85 blood pressure, 22 skin thickness, 200 insulin, 30 BMI, 0.3 diabetes pedigree, 55 age.
pp=np.array([[2,150,85,22,200,30,0.3,55]])
scaler.fit_transform(pp)
print(model.predict(scaler.fit_transform(pp)))