#import dataset
stat<-read.csv("C:\\Users\\jeeva\\Downloads\\R assignment\\Neural Netwrk\\50_Startups.csv")
#copy datas to another variable
stat1<-stat
#change to numerical datas
stat1$State<-as.numeric(stat1$State)
attach(stat1)
#newyork-3 california-1 florida-2

#normalising all column values
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}
stat_norm <- as.data.frame(lapply(stat1, normalize))
summary(stat_norm)
#Split datas to train and test datas
stat_train=stat_norm[1:40,]
stat_test=stat_norm[41:50,]
#import neuralnetwork
library(neuralnet)
#model prepared by neural net method for train data
model_stat=neuralnet(formula=Profit~R.D.Spend+Marketing.Spend+Administration+State,data=stat_train)
#plot the model 
plot(model_stat)
#the plot function which tells weight associated with each variable
#resulting the model with model and test datas
result_model<-compute(model_stat,stat_test[1:4])
str(result_model)#summarise result model
pre_stre<-result_model$net.result #predict model with output of summarised model
cor(pre_stre,stat_test$Profit)#correlation between predicted profit with test profit
#0.689

######Improve Model with many hidden values#######
model_stat2=neuralnet(Profit~R.D.Spend+Marketing.Spend+Administration+State,data=stat_train,hidden = 16)
plot(model_stat2)
#plot the model with hidden layers
#Resulting the model with improved model with test datas
result_model2<-compute(model_stat2,stat_test[1:4])
str(result_model2)#summarise result model
pre_stre2<-result_model2$net.result #predict model with output of summarised model
cor(pre_stre2,stat_test$Profit)#correlation between predicted profit with test profit
#0.7259873/10hidden #0.7505/15hidden  #0.8515/16hidden #0.572/17hiddend
#0.9095/16hidden Answer
#in 16 or lessthan 16 hidden layers they formed neural network graph properly
#In 17 hidden layers used they are not formed neural network.there gives some Nan Values.