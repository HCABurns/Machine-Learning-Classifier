#-------------------- Required imports --------------------#
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def knn(X,y):
    #Create a training / test split of the data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    #random_state = 1 - This makes the code consistent

    #Create K-Nearest-Neighbour classifer object
    clf = KNeighborsClassifier(3)

    #Train K-Nearest-Neighbour Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    #Determine the acuracy of the classifier.
    total = 0
    equal = 0
    for i,j in zip(y_pred,y_test):
        total+=1
        if i==j:
            equal +=1
        print(i , j , i==j, (equal/total))
        
    #Accuracy metric using sklearn
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    return round(equal/total,2)


if __name__ == "__main__":
    #Read the name file for aquiring the correct column names.
    #with open("iris.names") as f:
        #print(f.read())

    #Read in the file
    filename = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    #Set the column names
    names = ["Sepal Length","Sepal Width","Petal length","Petal Width","Class"]
    #Dictionary to map strings to values
    mapper = {'Iris-setosa':0, 'Iris-versicolor':1,'Iris-virginica':2}
    #Create dataframe
    df = pd.read_csv(filename , names = names)
    #Map the strings to the correct values.
    df["Class"] = df["Class"].map(mapper)
    #Testing print
    #print(df.head())

    #Define features and class variable.
    X = df[names] #Features
    y = df.Class #Define class variable

    accuracy = knn(X,y)
    print(f"Accuracy: {accuracy}")
    
    
