# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.

2.Analyse the data.

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result.


## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Jeyabalan
RegisterNumber:212222240040

import chardet
file = '/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd 
data = pd.read_csv("/content/spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy  
```

## Output:

## Result:
![image](https://github.com/jeyaqbalan7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393851/3d6ac155-7949-4982-98a2-183db9987c5c)

## Data.head():
![image](https://github.com/jeyaqbalan7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393851/46986cd4-f950-4dc8-82a6-8f783fc546bd)

## data.info():
![image](https://github.com/jeyaqbalan7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393851/0d7fdcf0-df32-4ca0-a1ad-fe5accd09ddd)

## data.isnull().sum():
![image](https://github.com/jeyaqbalan7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393851/e64d57b2-0592-4a78-8835-6abb37d48bea)

## Y prediction value:
![image](https://github.com/jeyaqbalan7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393851/1a898fb6-836c-4079-b510-57b6ce5769fe)

## Accuracy value:
![image](https://github.com/jeyaqbalan7/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393851/128363ca-5d75-4871-9fb8-c51c26c289a3)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
