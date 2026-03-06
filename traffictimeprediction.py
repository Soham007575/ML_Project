import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("indian_smart_traffic_dataset_pro.csv")

df_copy=df.copy()

df_copy["Total_Vehicles"] = df_copy[["Truck_Count",'Bike_Count','Car_Count']].sum(axis=1)


from sklearn.model_selection import train_test_split
x=df_copy[["Total_Vehicles"]]
y=df_copy['Signal_Waiting_Time_Seconds']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train,y_train)
y_pred = regression.predict(x_test)

Vehicles=int(input('Enter no of vehicle to predict the waiting time of traffic signal:-'))
prediction=regression.predict([[Vehicles]])[0]
print(f"Predicted time of waiting for vehicles at signal is {prediction:.2f}sec ")

