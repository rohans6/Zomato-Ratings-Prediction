# Loading Libraries
import pandas as pd 
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import ast
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,RobustScaler
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso,Ridge,ElasticNet
from sklearn.compose import ColumnTransformer
import pickle

# Loading data
dataset=pd.read_csv("zomato.csv")

#Data Cleaning
dataset.drop(['url','address','phone','name'],axis=1,inplace=True)
dataset.drop_duplicates(inplace=True)
dataset=dataset[(dataset['rate']!='NEW')&(dataset['rate']!='-')]
dataset['rate']=dataset['rate'].astype(str)
dataset['rate']=dataset['rate'].apply(lambda x: x.replace("/5",""))
dataset['rate']=dataset['rate'].apply(lambda x: float(x))
dataset=dataset.rename(columns={"approx_cost(for two people)":"cost","listed_in(type)":"type","listed_in(city)":'city'})
dataset['online_order']=dataset['online_order'].replace({"Yes","No"},{1,0})
dataset['book_table']=dataset['book_table'].replace({"Yes","No"},{1,0})
dataset['cost']=dataset['cost'].astype(str)
dataset['cost']=dataset['cost'].apply(lambda x: x.replace(",",""))
dataset['cost']=dataset['cost'].apply(lambda x: float(x))


# Dropping Missing Values
dataset=dataset.dropna()

# Preparing Data for Modelling
lb1=LabelEncoder()
dataset.location=lb1.fit_transform(dataset.location)
with open("Location_Encoder.pkl","wb") as f:
    pickle.dump(lb1,f)
lb2=LabelEncoder()
dataset.rest_type=lb2.fit_transform(dataset.rest_type)
with open("Rest_type_Encoder.pkl","wb") as f:
    pickle.dump(lb2,f)


X=dataset.iloc[:,[0,1,3,4,5,8]]
y=dataset['rate']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Training Model
lgm=ExtraTreesRegressor(n_estimators=1450)
lgm.fit(X_train,y_train)

with open("Model.pkl","wb") as f:
    pickle.dump(lgm,f)
