#read a dataset
starf<-read.csv("C:\\Users\\jeeva\\Downloads\\R assignment\\multi\\50_Startups.csv")
#change Categorical column to numeric
starf$State=as.numeric(factor(starf$State))
attach(starf)#attach the dataset
#Pair plots with each variabls
pairs(starf)
#correlation analysis of dataset
cor(starf)
#model Preparation
model.staf=lm(Profit~R.D.Spend+Administration+Marketing.Spend+State)
summary(model.staf)#
#p-value>0.05 , R-squared:  0.9507,	Adjusted R-squared:  0.9464 

#install package corpcor
install.packages("corpcor")
library(corpcor)#import corpcor

## Individual models for p value higher than 0.05 
model.staf1=lm(Profit~Administration)
summary( model.staf1)#
#p=0.1622>0.05 R-squared:  0.04029,	Adjusted R-squared:  0.02029 

model.stf2=lm(Profit~Marketing.Spend)
summary(model.stf2)# 
#p=4.38e-10<0.05 R-squared:  0.5592,	Adjusted R-squared:   0.55 

model.stf3=lm(Profit~State)
summary(model.stf3)#
#p=0.482>0.05  R-squared:  0.01036,	Adjusted R-squared:  -0.01025 

#model for p value high than 0.05 in individual analysis
model.stf4=lm(Profit~Administration+State)
summary(model.stf4)#
#administration(p-value)=0.167<0.05  State(p-value)=0.999>0.05
#R-squared:  0.0501,	Adjusted R-squared:  0.0097

###Partial correlation###
install.packages("corpcor")
library(corpcor)
cor2pcor(cor(starf))

#package car
library(car)
plot(model.staf)#plot the model identify outliers 
#influence factors
influence.measures(model.staf)
influenceIndexPlot(model.staf,id.no=3)
influencePlot(model.staf)
#50,49,46 are outliers

#deleting a outlier and predict model
model.staff=lm(Profit~R.D.Spend+Administration+State+Marketing.Spend,data=starf[-50,])
model.staff=lm(Profit~R.D.Spend+Administration+State+Marketing.Spend,data=starf[-49,])
model.staff=lm(Profit~R.D.Spend+Administration+State+Marketing.Spend,data=starf[-46,])
summary(model.staff)#now also P value has higher than 0.05
#R-squared:  0.9537,	Adjusted R-squared:  0.9495

#p -value >0.05 so we go Variance inflation factor
vif(model.staf)
#Variance Inflation Factor for input variables 
VIFRD=lm(R.D.Spend~Marketing.Spend+State+Administration)
VIFMS<-lm(Marketing.Spend~Administration+State+R.D.Spend)
VIFAD<-lm(Administration~Marketing.Spend+State+R.D.Spend)
VIFST<-lm(State~R.D.Spend+Marketing.Spend+Administration)
summary(VIFRD)## #state p value>0.05
summary(VIFMS)## #state p value>0.05
summary(VIFAD)## #state p value>0.05
summary(VIFST)## #other all p values>0.05 

#identify avplots for inputs are distributed with output are normal
avPlots(model.staf, id.n=2, id.cex=0.8, col="red")

#import Mass Package
install.packages("MASS")
library(MASS)

#find less models with StepAIC method
stepAIC(model.staf)

#identifies less models and we try to predict new models
model.f<-lm(Profit~R.D.Spend+Marketing.Spend)
summary(model.f)#
#pvalue<0.05 ,R-squared:  0.9507,	Adjusted R-squared:  0.9475 

#deleting outliers of data and to find high R and Adjusted R values 
model.ff<-lm(Profit~R.D.Spend+Marketing.Spend,data=starf[-50,])
model.ff<-lm(Profit~R.D.Spend+Marketing.Spend,data=starf[-49,])
model.ff<-lm(Profit~R.D.Spend+Marketing.Spend,data=starf[-46,])
summary(model.ff)#
#pvalue<0.05 ,R-squared:  0.9531,	Adjusted R-squared:  0.9511 
