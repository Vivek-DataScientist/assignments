
import numpy as np
import pandas as pd
airline= pd.read_csv("C:\\Users\\jeeva\\Downloads\\R assignment\\forecasting\\Airlines1.csv")
#month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 

p = airline["Month"][0]
p[0:2]
airline['month']= 0

#set Airline month values (1-12) for 96 obs 
for i in range(96):
    p = airline["Month"][i]
    airline['month'][i]= p[0:2]
    
#replace month values to month names
airline["month"]=airline["month"].replace([1],'Jan')
airline["month"]=airline["month"].replace([2],'Feb')
airline["month"]=airline["month"].replace([3],'Mar')
airline["month"]=airline["month"].replace([4],'Apr')
airline["month"]=airline["month"].replace([5],'May')
airline["month"]=airline["month"].replace([6],'Jun')
airline["month"]=airline["month"].replace([7],'Jul')
airline["month"]=airline["month"].replace([8],'Aug')
airline["month"]=airline["month"].replace([9],'Sep')
airline["month"]=airline["month"].replace([10],'Oct')
airline["month"]=airline["month"].replace([11],'Nov')
airline["month"]=airline["month"].replace([12],'Dec')

month_dummies = pd.DataFrame(pd.get_dummies(airline['month'])) #get dummies for 12months
airline = pd.concat([airline,month_dummies],axis = 1)##concat dummies with airline dataset 
#assign T values as 1 to 97 because 97 observations
airline["t"] = np.arange(1,97)
#assign T-square values
airline["t_squ"]=airline["t"]*airline["t"]
#get log values of passengers
airline["log"]=np.log(airline["Passengers"])
airline.Passengers.plot() #plotting the airline passengers observations

#split the datas to train and test
Train=airline.head(70)
Test=airline.tail(16)
####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear
# 50.116532625114495
##################### Exponential ##############################

Exp = smf.ols('log~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp
#44.60
#################### Quadratic ###############################

Quad = smf.ols('Passengers~t+t_squ',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squ"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad
# 44.45799618474633 compare to linear and exponential this is less rmse value
################### Additive seasonality ########################
add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea
#139.53794884068864 #compare to multiple seasonality additive seasonality rmse values is less
################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Passengers~t+t_squ+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squ']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 
#33.633  #additive seasonality with quadratic trend the RMSE values are less
################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea
#144.17574
###############test#################
3  #additive seasonality with quadratic trend the RMSE values are less
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
#Print the tables od RMSE values