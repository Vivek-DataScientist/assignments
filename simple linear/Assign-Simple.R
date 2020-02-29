############calories#########
new=read.csv(file.choose())
names(new)[1]="weight"
names(new)[2]="calories"
new
attach(new)
library(lattice)
dotplot(weight,main="Dotplot for grams")
dotplot(calories,main="dotplot for calories")
boxplot(weight)
boxplot(calories)
hist(weight)
hist(calories)
qqnorm(weight)
qqline(weight)
qqnorm(calories)
qqline(calories)
hist(weight)
lines(density(weight))
lines(density(weight,adjust=2))
hist(calories)
lines(density(calories))
lines(density(calories,adjust=2))
plot(weight,calories)
attach(new)

cor(weight,calories)#0.946991(Positive correlation)
regg<-lm(calories~weight)
summary(regg)#R=0.8968,Adjusted R-squared:  0.8882 
sqrt(sum(regg$residuals^2)/nrow(new))#232.8335

##############Transformation technique#####
##output with square of input variable###
cor(weight,sqrt(calories)) #0.9255962 (Positive correlation)
regg_sqrt<-lm(weight~sqrt(calories),data=new)
summary(regg_sqrt)#R= 0.8567 Adjusted R-squared: 0.8448 
sqrt(sum(regg_sqrt$residuals^2)/nrow(new))#121.71

##############Transformation technique#####
##output with log of input variable###
cor(weight,log(calories))#0.8987253(Positive correlation)
regg_log<-lm(weight~log(calories),data=new)
summary(regg_log)#R=0.8077,Adjusted R-squared:  0.7917 
sqrt(sum(regg_log$residuals^2)/nrow(new))#141.0054

##############Transformation technique#####
##output square with log of input variable###
cor(log(calories),weight*weight)#0.8249636
reg11<-lm(log(calories)~weight+I(weight*weight),data = new)
summary(reg11)#R=0.8516,Adjusted R-squared:0.8246 
sqrt(sum(reg11$residuals^2))/nrow(new)#0.03137703

##Square of output with input variable###
cor(calories,sqrt(weight))#0.9559736
regg_sqrt1=lm(calories~sqrt(weight),data=new)
summary(regg_sqrt1) #R=0.9139,Adjusted R-squared:0.9067
sqrt(sum(regg_sqrt1$residuals^2)/nrow(new))#212.68
###Prediction model for high R & adjusted R value ####
pred6=as.data.frame(predict(regg_sqrt1,interval="predict"))
Absolute=abs(pred6$fit)
errorr=Absolute-new$weight#with output
sqrt(sum(errorr^2)/nrow(new))#2018.756
plot(Absolute,errorr)

##log of output with input variable###
cor(log(weight),calories)#0.936
regg_log1=lm(log(weight)~calories,data=new)
summary(regg_log1)#R=0.877,Adjusted R-squared:  0.8674

