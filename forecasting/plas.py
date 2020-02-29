
#importing packages
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # Simple Exponential Smoothing
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing #Exponential smoothing

#read a dataset
plas=pd.read_csv("C:\\Users\\jeeva\\Downloads\\R assignment\\forecasting\\PlasticSales.csv")
p = plas["Month"][0]
p[0:3]
plas['month']= 0

#divide month names for 60 obs 
for i in range(60):
    p = plas["Month"][i]
    plas['month'][i]= p[0:3]

month_dummies = pd.DataFrame(pd.get_dummies(plas['month'])) #get dummies for 12months
plas = pd.concat([plas,month_dummies],axis = 1)##concat dummies with plastic dataset 
#assign T values as 1 to 60 because 60 observations
plas["t"] = np.arange(1,61)
#assign T-square values
plas["t_squ"]=plas["t"]*plas["t"]
#get log values of passengers
plas["log"]=np.log(plas["Sales"])
plas.Sales.plot()

#split the datas to train and test
Train=plas.head(48)
Test=plas.tail(12)

####################### LINEAR ##########################
import statsmodels.formula.api as smf 
linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear
#260.93781
##################### Exponential ##############################

Exp = smf.ols('log~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp
#268.6938
#################### Quadratic ###############################

Quad = smf.ols('Sales~t+t_squ',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squ"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad
#297.4067

################### Additive seasonality ########################
add_sea = smf.ols('Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea
#235.602
################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Sales~t+t_squ+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squ']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 
#218.1938 
################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea
# 239.6543

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)

# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Sales)
#17.0415
#compare to many method simple exponential method gives less error

# Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Sales)
#101.995

# simple exponential smoothing with additive seasonality and additive trend
sme_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
pred_sme_add_add = sme_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_sme_add_add,Test.Sales)
#14.74555

# simple exponential smoothing with multiplicative seasonality and additive trend
mst_model_mul_add = ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_mst_mul_add = mst_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_mst_mul_add,Test.Sales)# 15.00214

# simple exponential smoothing with additive seasonality and additive trend gives less error ...so we go this method
