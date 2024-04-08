# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas, numpy and mathplotlib.pyplot.
2. Trace the best fit line and calculate the cost function.
3. Calculate the gradient descent and plot the graph for it.
4. Predict the profit for two population sizes. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Kanimozhi
RegisterNumber:  212222230060
*/
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linearREG(X1,Y,learnRate=0.01,Iteration=1000):        # Add a column of ones to X for the intercept term
    X=np.c_[np.ones(len(X1)),X1]                          # Initialize theta with zeros
    theta=np.zeros(X.shape[1]).reshape(-1,1)              # Perform gradient descent
    for _ in range(Iteration):
        predictions=(X).dot(theta).reshape(-1,1)          # Calculate predictions
        errors=(predictions-Y).reshape(-1,1)              # Calculate errors
        theta-=learnRate*(1/len(X1))*X.T.dot(errors)      # Update theta using gradient descent
    return theta
data=pd.read_csv('CSVs/50_Startups.csv',header=None)
print(data.head())
X=(data.iloc[1:,:-2].values) # Assuming the last column is your target variable 'Y' and the preceding column
print(X)
X1=X.astype(float)
scaler=StandardScaler()
Y=(data.iloc[1:,-1].values).reshape(-1,1)
print(Y)
X1scaled=scaler.fit_transform(X1)
Y1scaled=scaler.fit_transform(Y)
print(X1scaled,Y1Scaled)
theta=linearREG(X1scaled,Y1scaled)                             # Learn model parameters
newData=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)   # Predict target value for a new data point
newScaled=scaler.fit_transform(newData)
prediction=np.dot(np.append(1, newScaled), theta) 
prediction=prediction.reshape(-1,1) 
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")

## Output:
![image](https://github.com/kanimozhipannerselvam/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119476060/75737b78-713e-4b97-a04a-cb616b505646)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
