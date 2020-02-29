#importing pckages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # Simple Exponential Smoothing
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing #Exponential smoothing
import statsmodels.graphics.tsaplots as tsa_plots

#read a dataset
coca=pd.read_csv("C:\\Users\\jeeva\\Downloads\\R assignment\\forecasting\\CocaCola1.csv")
#divide a quarters and years
coca['qr']=coca.Quarter.str.slice(-5,-3).astype(str)
coca['yr']=coca.Quarter.str.slice(3,5).astype(int)
#based on quarter we plot the sales values
heatmap_y_year = pd.pivot_table(data=coca,values="Sales",index="yr",columns="qr",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_year,annot=True,fmt="g")

# Boxplot base with quarter and year
sns.boxplot(x="qr",y="Sales",data=coca)
sns.boxplot(x="yr",y="Sales",data=coca)
# moving average for the time series to understand better about the trend character in coca
coca.Sales.plot(label="org")
for i in range(2,24,6):
    coca["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)
    
# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(coca.Sales,model="additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(coca.Sales,model="multiplicative")
decompose_ts_mul.plot()

# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(coca.Sales,lags=10)
tsa_plots.plot_pacf(coca.Sales)

# splitting the data into Train and Test data and considering the last 4 quarter data as 
# Test data and left over data as train data 

Train =coca.head(30)
Test = coca.tail(12)
# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13),inplace=True)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)

# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Sales) #  16.642062245135346

# Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Sales) #  : 8.998775407006265
#compare to simple exponential holt method  give less error 

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=4,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Sales) # 8.412401039231478

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Sales)# 4.333179820930876

#plotting different plots
plt.plot(Train.index, Train["yr"], label='Train',color="black")
plt.plot(Test.index, Test["yr"], label='Test',color="blue")
plt.plot(pred_ses.index, pred_ses, label='SimpleExponential',color="green")
plt.plot(pred_hw.index, pred_hw, label='Holts_winter',color="red")
plt.plot(pred_hwe_add_add.index,pred_hwe_add_add,label="HoltsWinterExponential_1",color="brown")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="HoltsWinterExponential_2",color="yellow")

plt.legend(loc='best')