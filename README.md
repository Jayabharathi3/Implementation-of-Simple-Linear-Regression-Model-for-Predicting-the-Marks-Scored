# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: JAYABHARATHI.S
RegisterNumber:  212222100013
*/
```
```python

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())

#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred =reg.predict(X_test)
print(Y_pred)
print(Y_test)

#Graph plot for training data
plt.scatter(X_train,Y_train,color='blue')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:

 # 1. df.head()
 ![image](https://github.com/Jayabharathi3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120367796/26c4bfb0-0e5b-49e1-87fd-6b754027ddab)

 
 # 2. df.tail()
 ![image](https://github.com/Jayabharathi3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120367796/18485cf5-edeb-47aa-8681-89519d735c72)

    
 # 3. Array value of X
 ![image](https://github.com/Jayabharathi3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120367796/8f8b89cc-7dc2-4c94-89a9-c7a1ef2e8313)


 # 4. Array value of Y
![image](https://github.com/Jayabharathi3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120367796/670cdc76-9c84-4ef9-a007-7a823beb9803)

 
 # 5. Values of Y prediction
![image](https://github.com/Jayabharathi3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120367796/fe13973c-2bd5-48c4-947b-df3be00b2e0c)

 
 # 6. Array values of Y test
![image](https://github.com/Jayabharathi3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120367796/36c77ddd-5696-4a18-9928-05c54c294b0e)

 
 # 7. Training Set Graph
![training graph](https://github.com/Jayabharathi3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120367796/d98b6194-0129-447f-8868-7bbf22a804bf)

 
 # 8. Test Set Graph
![mlexp2](https://github.com/Jayabharathi3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120367796/1992bf3e-c215-4719-a7b5-2f0ab2fc1d5a)

 
 # 9. Values of MSE, MAE and RMSE
 ![image](https://github.com/Jayabharathi3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120367796/5b8148b6-f7fb-4385-a5ee-06088509e8dd)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
