emp<-read.csv(file.choose())
names(emp)[1]="salary" # change column names 
names(emp)[2]="churn"
attach(emp)

cor(churn,salary)##-0.9117216
regaa=lm(churn~salary)
summary(regaa)## R-squared:  0.8312,	Adjusted R-squared:  0.8101 
sqrt(sum(regaa$residuals^2)/nrow(emp))# 3.997528
#output::Churn input::salary 
#input with sqrt
cor(churn,sqrt(salary))##-0.9165311(Negative correlation)
regbb=lm(churn~sqrt(salary))
summary(regbb)##R-squared:   0.84,	Adjusted R-squared:   0.82  
sqrt(sum(regbb$residuals^2)/nrow(emp))#3.891995

#input with log
cor(churn,log(salary))##-0.9212077(Negative correlation)
regcc=lm(churn~log(salary))
summary(regcc)##  R-squared:  0.8486,	Adjusted R-squared:  0.8297 
sqrt(sum(regcc$residuals^2)/nrow(emp))#3.786

#output on sqrt with input on log
cor(log(salary),churn*churn)## -0.8965286(Negative correlation)
regdd=lm(log(salary)~churn+I(churn*churn))
summary(regdd)## R-squared:  0.9786,	Adjusted R-squared:  0.9725 
sqrt(sum(regdd$residuals^2)/nrow(emp))#0.0074
########## Prediction for Better R values ########
predg=as.data.frame(predict(regdd,interval = "predict"))
Expo=exp(predg$fit)
error=Expo-churn
sqrt(sum(err^2)/nrow(emp))# 66.19578
plot(Expo,error)

###output Variable sqrt
cor(salary,sqrt(churn))##-0.9235755(Negative correlation)
regee=lm(sqrt(churn)~salary)
summary(regee)##R-squared:0.853,	Adjusted R-squared:  0.8346
sqrt(sum(regee$residuals^2)/nrow(emp))#0.215


###output Variable log
cor(salary,log(churn))#-0.934(Negative correlation)
regff=lm(log(churn)~salary)
summary(regff)# R-squared:  0.8735,	Adjusted R-squared:  0.8577
sqrt(sum(regff$residuals^2)/nrow(emp))#0.04
