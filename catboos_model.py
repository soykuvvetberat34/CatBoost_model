from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd
import time 
import numpy as np
time1=time.time()
datas=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Turkcell Makine Öğrenmesi\\Hitters.csv")
datas=datas.dropna()
dms=pd.get_dummies(datas[["League","Division","NewLeague"]])
y=datas["Salary"]
x_=datas.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
x=pd.concat([x_,dms],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

cbr=CatBoostRegressor()
cbr_params={
    "iterations":[200,500,1000],#iterations fit edilecek ağac sayısı
    "learning_rate":[0.01,0.1],
    "depth":[3,6,8]
}
cbr_cv=GridSearchCV(cbr,cbr_params,cv=10,n_jobs=-1,verbose=2)
cbr_cv.fit(x_train,y_train)
iterations=cbr_cv.best_params_["iterations"]
learning_rate=cbr_cv.best_params_["learning_rate"]
depth=cbr_cv.best_params_["depth"]
cbr_tuned=CatBoostRegressor(depth=depth,learning_rate=learning_rate,iterations=iterations)
cbr_tuned.fit(x_train,y_train)
predict=cbr_tuned.predict(x_test)
rmse=np.sqrt(mean_squared_error(y_test,predict))
time2=time.time()
print(time2-time1)
print(rmse)




