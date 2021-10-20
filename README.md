# KNN-classification-Practice
#Write Python code in PyCharm to perform a KNN classification 
#Create a DataFrame “diabetes_knn” to store the diabetes data.
#Determine the dimensions of the “diabetes_knn” dataframe.
#Update the DataFrame to account for missing values.
#No missing values
#Create the Feature Matrix (X) and Target Vector (y).
#Standardize the attributes of Feature Matrix (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
#Split the Feature Matrix and Target Vector into three partitions. Training A, Training B and test. They should be in the ratio 60-20-20. random_state = 2021, #stratify = y
#Develop a KNN based model based on Training A for various ks. K should range between 1 and 30.
#Compute the KNN score (accuracy) for training A and training B data for those ks.
#Plot a graph of training A and training B accuracy and determine the best value of k. Label the plot.
#Now, using the selected value of k, score the test data set k=14
#Plot the confusion matrix (as a figure).
#Predict the Outcome for a person with 2 pregnancies, 150 glucose, 85 blood pressure, 22 skin thickness, 200 insulin, 30 BMI, 0.3 diabetes pedigree, 55 age.
