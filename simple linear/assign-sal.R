#read a salary dataset 
sala<-read.csv(file.choose())
names(sala)[1]="year" #change column names
names(sala)[2]="salary"
attach(sala)#attaching the dataset

#correlation Analysis
cor(salary,year)#0.9782416
regv<-lm(salary~year)#linear model for dataset
summary(regv)#R-squared:  0.957,	Adjusted R-squared:  0.9554 
sqrt(sum(regv$residuals^2)/nrow(sala))#RMSE : 5592.044

#correlation Analysis with transformation technique
cor(salary,sqrt(year))#0.96488
regi<-lm(salary~sqrt(year))#linear model with squared technique
summary(regi)#R-squared:  0.931,	Adjusted R-squared:  0.9285 
sqrt(sum(regi$residuals^2)/nrow(sala))#RMSE : 7080.096

#correlation Analysis with transformation technique
cor(salary,log(year))# 0.9240611
regve<-lm(salary~log(year))#linear model with log technique
summary(regve)# R-squared:  0.8539,	Adjusted R-squared:  0.8487 
sqrt(sum(regve$residuals^2)/nrow(sala))#RMSE :: 10302.89

#correlation Analysis with transformation technique
cor(log(salary),year)# 0.9653844
regvk<-lm(log(salary)~year)#linear model with log technique on Output variable
summary(regvk)# R-squared:  0.932,	Adjusted R-squared:  0.9295 
sqrt(sum(regvk$residuals^2)/nrow(sala))#RMSE :: 0.09457437

#correlation Analysis with transformation technique
cor(sqrt(salary),year)# 0.975
regvu<-lm(sqrt(salary)~year)#linear model with sqrt technique on Output variable
summary(regvu)#  R-squared:  0.9498,	Adjusted R-squared:  0.948  
sqrt(sum(regvu$residuals^2)/nrow(sala))#RMSE :: 10.93
#prediction model for Better R values
predd<-as.data.frame(predict(regvu,interval = "predict"))
Absolute=abs(predd$fit)#Find absolute value because sqrt technique gives high R values
errore=Absolute-sala$year# find error with year
plot(Absolute,errore)#plot the graph
