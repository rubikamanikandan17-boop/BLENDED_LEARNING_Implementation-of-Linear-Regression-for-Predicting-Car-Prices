# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: Rubika
RegisterNumber: 212225040348
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
df=pd.read_csv('CarPrice_Assignment.csv')
df.head
X=df[['enginesize','horsepower','citympg','highwaympg']]
Y=df['price']
df.head()
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
#feature scaling
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
#train model
model=LinearRegression()
model.fit(X_train_scaled,Y_train)
#prediction
Y_pred=model.predict(X_test_scaled)
print("Name:Rubika")
print("Reg. No:212225040348")
print("MODEL COEFFICIENTS:")
for feature,coef in zip(X.columns,model.coef_):
    print(f"{feature:>12}: {coef:>10}")
print(f"{'Intercept':>12}: {model.intercept_:>10}")
print("\nMODEL PERFORMANCE:")
print(f"{'MSE':>12}: {mean_squared_error(Y_test,Y_pred):>10}")
print(f"{'RMSE':>12}: {np.sqrt(mean_squared_error(Y_test,Y_pred)):>10}")
print(f"{'R-squared':>12}: {r2_score(Y_test,Y_pred):10}")
print(f"{'MAE':>12}: {mean_absolute_error(Y_test,Y_pred):10}")
# linearity check
plt.figure(figsize=(10,5))
plt.scatter(Y_test,Y_pred,alpha=0.6)
plt.plot([Y.min(),Y.max()],[Y.min(),Y.max()],'r--')
plt.title("Linearity Check: Actual vs Predicted Price")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()
# Independence (Durbin-watson)
residuals=Y_test-Y_pred
dw_test=sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson statistic: {dw_test:.2f}",
      "\n(Values close to 2 indicates no autocorrelation)")
# Homoscedasticity
plt.figure(figsize=(10,5))
sns.residplot(x=Y_pred,y=residuals,lowess=True,line_kws={'color':'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()
# Normality of residuals
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
sns.histplot(residuals,kde=True,ax=ax1)
ax1.set_title("Residuals Distrubution")
sm.qqplot(residuals,line='45',fit=True,ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: 
RegisterNumber:  
*/
```

## Output:
Name: Rubika
Reg. No: 212225040348
MODEL COEFFICIENT:
  enginesize: 4523.404901011966
  horsepower: 1694.2232554525806
     citympg: -392.5731841571549
  highwaympg: -816.3577991826088
   Intercept: 13223.414634146342
df = pd.read_csv('CarPrice_Assignment.csv')

\MODEL PERFORMANCE:
         MSE: 16471505.900042146
        RMSE: 4058.5103055237087
   R-squared: 0.7913520781370976

   <img width="868" height="468" alt="image" src="https://github.com/user-attachments/assets/2d2a26e5-7a19-4317-841e-a1f8e8cdfd15" />

   Durbin-Watson Statistic: 2.28 
(Values close to 2 indicate no autocorrelation)

<img width="880" height="468" alt="image" src="https://github.com/user-attachments/assets/a186d4df-fea1-4c33-9acb-f56d8ae7724a" />

<img width="880" height="468" alt="image" src="https://github.com/user-attachments/assets/135635f6-56ca-4a87-ac09-8612cb50513d" />
<img width="997" height="468" alt="image" src="https://github.com/user-attachments/assets/b7a4db5f-a5db-4868-b218-bdf31ec0de00" />




![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
