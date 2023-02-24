

## Importing the Dependecies 
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer



## Data Colection and Preprocessing
data = pd.read_csv("/kaggle/input/spam-mail-detection-dataset/spam_mail_data.csv")

data.isnull().sum()

data.replace(to_replace=["ham","spam"], value=[0,1], inplace=True)

x = data.drop(["Category"],axis=1)

x = x.to_numpy()

y = data["Category"]

x = x.reshape(-1)



## Converting data from text to numerical
vt = TfidfVectorizer(stop_words='english', min_df=1)
x = vt.fit_transform(x)



## Splitting the Data
xtn,xtt,ytn,ytt = train_test_split(x,y, test_size=0.05, random_state=1 )

## Training The Model
model1 = XGBClassifier()
model2 = LogisticRegression()

model1.fit(xtn,ytn)
model2.fit(xtn,ytn)



## Model Evaluation Through MSE
y_pred1 = model1.predict(xtt)
y_pred2 = model2.predict(xtt)

acc_score1 = accuracy_score(ytt,y_pred1)
acc_score2 = accuracy_score(ytt,y_pred2)


print(f"The accuracy score of XGBClassifier is {acc_score1}")
print(f"The accuracy score of logistic regression model is {acc_score2}")


