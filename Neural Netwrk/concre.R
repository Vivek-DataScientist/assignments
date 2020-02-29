#load the concrete data
concrete = read.csv("C:\\Users\\jeeva\\Downloads\\R assignment\\Neural Netwrk\\concrete.csv")
#normalization function
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

# apply normalization to entire data frame
concrete_norm <- as.data.frame(lapply(concrete, normalize))
summary(concrete_norm)
# Split training and test data
concrete_train <- concrete_norm[1:773, ]
concrete_test <- concrete_norm[774:1030, ]

## Training a model on the data ----
# train the neuralnet model
library(neuralnet)

# model prepared by neural network method on train data
concrete_model <- neuralnet(formula = strength ~ cement + slag +
                              ash + water + superplastic + 
                              coarseagg + fineagg + age,
                            data = concrete_train)


# visualize the network topology
plot(concrete_model)

## Evaluating model performance 
# obtain model results
results_model <- compute(concrete_model, concrete_test[1:8])
# obtain predicted strength values
str(results_model)
predicted_strength <- results_model$net.result#predict model with output of summarised model
# correlation between predicted strength and test strength
cor(predicted_strength, concrete_test$strength)
#normal correlation without hidden layer 0.8063
## Improving model performance with hidden layers##
concrete_model2 <- neuralnet(strength ~ cement + slag +
                               ash + water + superplastic + 
                               coarseagg + fineagg + age,
                             data = concrete_train, hidden = 15)
# plot the model
plot(concrete_model2)
#model results after providing hidden layers 
model_results2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_results2$net.result ##predict model with output of summarised model
cor(predicted_strength2, concrete_test$strength)#correlation between predicted profit with test profit
#0.947/10 hidden
#9288/15 hidden