#read a dataset
forest<-read.csv("C:\\Users\\jeeva\\Downloads\\R assignment\\svm\\forestfires.csv")
#take columns from 3 to 31
forest1<-forest[,3:31]
#change output columns in 1st place
forest1<-forest1[,c(29,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28)]
#split into train and test
forest_train=forest1[1:399,]
forest_test=forest1[400:517,]

##Training a model on the data ----
# begin by training a simple linear SVM
library(kernlab)
#doing a Model with KSVM technique which kernel is vanniladot
forest_classifier <- ksvm(size_category ~ ., data = forest_train,kernel = "vanilladot")
forest_classifier#Training error : 0 support vectors : 12
#predict testing data with model
forest_predictions <- predict(forest_classifier, forest_test)
table(forest_predictions, forest_test$size_category)
agreement1 <- forest_predictions == forest_test$size_category
table(agreement1)#False 4 True 114
prop.table(table(agreement1))
#FALSE       TRUE 
#0.03389831 0.96610169 #high accuracy so we use this kernel

#improving model performance
#doing a model with ksvm technique where kernel is rbfdot
forest_classifier <- ksvm(size_category ~ ., data = forest_train,kernel = "rbfdot")
forest_classifier #Training error : 0.0.130 Number of Support Vectors : 201
#predict testing data with model
forest_predictions <- predict(forest_classifier, forest_test)
table(forest_predictions,forest_test$size_category)
agreement2 <- forest_predictions == forest_test$size_category
table(agreement2)
prop.table(table(agreement2))
#FALSE      TRUE 
#0.2881356 0.7118644  #less true accuracy so this is not correct

#improving model performance
#doing a model with ksvm technique where kernel is polydot
forest_classifier <- ksvm(size_category ~ ., data = forest_train,kernel = "polydot")
forest_classifier #Training error : 0 support vector=12
#predict testing data with model
forest_predictions <- predict(forest_classifier, forest_test)
table(forest_predictions,forest_test$size_category)
agreement2 <- forest_predictions == forest_test$size_category
table(agreement2)
prop.table(table(agreement2))
#FALSE       TRUE 
#0.03389831 0.96610169 #high true accuracy so we go polydot

#doing a model with ksvm technique where kernel is laplacedot
forest_classifier <- ksvm(size_category ~ ., data = forest_train,kernel = "anovadot")
forest_classifier #Training error :0  support vector=196
#predict testing data with model
forest_predictions <- predict(forest_classifier, forest_test)
table(forest_predictions,forest_test$size_category)
agreement2 <- forest_predictions == forest_test$size_category
table(agreement2)
prop.table(table(agreement2))
#FALSE      TRUE 
#0.1101695 0.8898305 #less accuracy compared with polydot..so we go for polydot