#read a delivery dataset
deliv=read.csv(file.choose())
names(deliv)[1]="delivery" #change column names
names(deliv)[2]="sort"
attach(deliv) #attach dataset
hist(sort) #histogram diagram for Sort Column
hist(delivery) #histogram diagram for delivery Column

#correlation analysis
cor(delivery,sort) ###0.8259
rega<-lm(delivery~sort)#linear model 
summary(rega)# R-squared:  0.6823,	Adjusted R-squared:  0.6655 
sqrt(sum(rega$residuals^2)/nrow(deliv))###2.7916

#correlation analysis with transformation technique
cor(delivery,sqrt(sort)) ###0.834
regb<-lm(delivery~sqrt(sort))#linear model with sqrt technique
summary(regb)#R-squared:  0.6958,	Adjusted R-squared:  0.6798 
sqrt(sum(regb$residuals^2)/nrow(deliv)) ###2.7315

#correlation analysis with transformation technique
cor(delivery,log(sort))###0.833
regc<-lm(delivery~log(sort))#Linear model with log technique
summary(regc)#R-squared:  0.6954,	Adjusted R-squared:  0.6794 
sqrt(sum(regc$residuals^2)/nrow(deliv))###2.7331

#correlation analysis with transformation technique
cor(log(delivery),I(sort*sort))###0.7882
regd<-lm(log(delivery)~sort-I(sort*sort))#linear model with log & sqrt technique
summary(regd)#R-squared:  0.7109,	Adjusted R-squared:  0.6957 
sqrt(sum(regd$residuals^2)/nrow(deliv))###0.1669

#correlation analysis with transformation technique
cor(sqrt(delivery),sort)###0.839
rege<-lm(sqrt(delivery)~sort)#linear model with sqrt technique
summary(rege)#R-squared:  0.704,	Adjusted R-squared:  0.6885 
sqrt(sum(rege$residuals^2)/nrow(deliv))#0.3323

#correlation analysis with transformation technique
cor(log(delivery),sort)###0.8431
regf<-lm(log(delivery)~sort)#linear model with log technique
summary(regf)#R-squared:  0.7109,	Adjusted R-squared:  0.6957 
sqrt(sum(regf$residuals^2)/nrow(deliv))#0.1669
#prediction model for Better R values
predw=as.data.frame(predict(regf,interval="predict"))
y=exp(predw$fit)#exponential technique because log gives high R values
err=y-deliv$sort #to find error with sort values
sqrt(sum(err^2)/nrow(deliv))####10.57
plot(y,err)#plotting the graph
