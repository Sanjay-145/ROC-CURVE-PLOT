### EX NO : 07
### DATE  : 06/05/2022
# <p align="center"> ROC CURVE PLOT </p>
## Aim:
   To write python code to plot ROC curve used in ANN.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner /Google Colab

## Related Theory Concept:
An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters:

True Positive Rate (TPR) is a synonym for recall and is therefore defined as follows:
![image](https://user-images.githubusercontent.com/75235426/169490815-45349c6b-2dfa-4c00-9a97-7ede6ec6d87f.png)


False Positive Rate (FPR) is defined as follows:
![image](https://user-images.githubusercontent.com/75235426/169490849-ba3a71b5-2e6c-4223-88e9-3f93da49f87b.png)

An ROC curve plots TPR vs. FPR at different classification thresholds. Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives. The following figure shows a typical ROC curve.

## Algorithm
1.Import the necessary libraries
2.Load the dataset and split the training and testing sets.
3.Fit the training set to the logistic regression model.
4.Display the results with the test data.

## Program:
```
/*
Program to plot Receiver Operating Characteristic [ROC] Curve.
Developed by   : P.Sanjay
RegisterNumber :  212220230042
*/
```
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
#import dataset from CSV file
data = pd.read_csv("default.csv")
#define the predictor variable and the response variable
X=data[['student','balance','income']]
y= data ['default']

#split the dataset into training(70%) and testing(30%) sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
#instantiate the model
log_regression=LogisticRegression()
#fit the model using the training data 
log_regression.fit(X_train,y_train)
#define the metrics
y_pred_proba= log_regression.predict_proba(X_test)[::,1]
fpr,tpr,_= metrics.roc_curve(y_test, y_pred_proba)

#create the ROC curve
plt.plot(fpr,tpr)
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

#define metrics
y_pred_proba=log_regression.predict_proba(X_test)[::,1]
fpr,tpr,_=metrics.roc_curve(y_test,y_pred_proba)
auc=metrics.roc_auc_score(y_test,y_pred_proba)

#create the ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.legend(loc=4)
plt.show()
```

## Output:
![image](https://user-images.githubusercontent.com/75235426/169492837-aa22e8e8-f4bf-4b31-b844-c37f575c875d.png)


## Result:
Thus the python program successully plotted Receiver Operating Characteristic [ROC] Curve.
