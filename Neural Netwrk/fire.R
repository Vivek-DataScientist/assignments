#read a dataset
forest<-read.csv("C:\\Users\\jeeva\\Downloads\\R assignment\\Neural Netwrk\\forestfires.csv")
# custom normalization function
# i take 3rd column to 31st column
forest1<-forest[3:31]
#change to numeric data
forest1$size_category<-as.numeric(forest1$size_category)
attach(forest1)
# small = 2 large = 1
#normalisation function
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}
# apply normalization to entire data frame
forest_norm <- as.data.frame(lapply(forest1, normalize))
summary(forest_norm)
# create training and test data
forest_train <- forest_norm[1:400, ]
forest_test <- forest_norm[401:517, ]

colnames(forest1)
## Training a model on the dat with neuralnet method
library(neuralnet)

# model prepared by neuralnet method with train datas
forest_model <- neuralnet(size_category ~ FFMC+DMC+DC+ISI+temp+RH+wind+rain+area+dayfri+daymon+daysat+    
                                          daysun+daythu+daytue+daywed+monthapr+monthaug+monthdec+monthfeb+monthjan+     
                                          monthjul+monthjun+monthmar+monthmay+monthnov+monthoct+monthsep,
                                               data=forest_train)
plot(forest_model)                
results_model <- compute(forest_model, forest_test[,1:28])
# resulting the model with model and test datas
str(results_model)
predicted_strength <- results_model$net.result #predict model with output of summarised model
# correlation between predicted strength and test strength
cor(predicted_strength, forest_test$size_category)
#0.8605507

## Improving model performance with hidden neurons
forest_model2 <- neuralnet(size_category ~ FFMC+DMC+DC+ISI+temp+RH+wind+rain+area+dayfri+daymon+daysat+    
                            daysun+daythu+daytue+daywed+monthapr+monthaug+monthdec+monthfeb+monthjan+     
                            monthjul+monthjun+monthmar+monthmay+monthnov+monthoct+monthsep,
                          data=forest_train,hidden=15)
plot(forest_model2)                
## Evaluating model performance 
# obtain model results
results_model2 <- compute(forest_model2, forest_test[,1:28])
str(results_model2)
predicted_strength2 <- results_model2$net.result#predict model with output of summarised model
# correlation between predicted strength and test strength
cor(predicted_strength2, forest_test$size_category)
# examine the correlation between predicted strength and test strength
#0.5974578/10hid we get low correlation
#0.08353834/1hid
#0.6595136/14hid 0.7055942/15hid 0.6103218/16hid
#0.7055942/15hid