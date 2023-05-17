#-------------------- Required imports --------------------#
import pandas as pd
import numpy as np
from decisionStump import decisionStump


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

    X = df[names[0:-1]]
    y = df.Class

    accuracy = decisionStump(X,y)
    print(accuracy)
    
