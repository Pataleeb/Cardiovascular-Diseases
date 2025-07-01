###Cardiovascular diseases 

cvd <- read.csv("cardio_train.csv",sep=";")
head(cvd)
cvd <- cvd[,-1]
colnames(cvd)
dim(cvd)

#Splitting the Dataset
set.seed(123)
flag <- sort(sample(70000,23100, replace = FALSE))
cvdtrain <- cvd[-flag,]
cvdtest <- cvd[flag,]
## Extract the true response value for training and testing data
y1    <- cvdtrain$cardio;
y2    <- cvdtest$cardio;

#Random Forest Model
library(caret)
library(randomForest)
library(mlbench)
set.seed(123)
# Define parameter grid for mtry, ntree, and nodesize
param_grid <- expand.grid(
  mtry = c(3, 6, 9, 12),  
  ntree = seq(50, 500, by = 50),  
  nodesize = seq(5, 20, by = 5) 
)
ctrl <- trainControl(
  method = "cv",
  number = 5
)
results <- list()
# Iterate over parameter grid
for (i in 1:nrow(param_grid)) {
  # Hyperparameter tuning
  final_rf_model <- train(
    as.factor(cardio) ~ .,
    data = cvdtrain,
    method = "rf",
    trControl = ctrl,
    tuneGrid = data.frame(mtry = param_grid$mtry[i]),  # Selecting mtry value
    ntree = param_grid$ntree[i],  # Setting ntree
    nodesize = param_grid$nodesize[i]  # Setting nodesize
  ) 
  results[[i]] <- final_rf_model
}
best_model_index <- which.max(sapply(results, function(x) x$results$Accuracy))
best_model <- results[[best_model_index]]
best_mtry <- param_grid$mtry[best_model_index]
best_ntree <- param_grid$ntree[best_model_index]
best_nodesize <- param_grid$nodesize[best_model_index]
# Print the best hyperparameters
cat("Best mtry:", best_mtry, "\n")
cat("Best ntree:", best_ntree, "\n")
cat("Best nodesize:", best_nodesize, "\n")

final_rf_model <- randomForest(as.factor(cardio) ~ ., data = cvdtrain, 
                               ntree = 500, mtry = 3, nodesize = 15, importance=TRUE)
importance(final_rf_model)
importance(final_rf_model, type=2)
varImpPlot(final_rf_model)

#Logistic Regression Model
library(glmnet)
# Fit initial logistic regression model
mod4 <- glm(cardio ~ ., family = binomial, data = cvdtrain)
mod4_step <- step(mod4)
summary(mod4_step)
pred_train <- predict(mod4_step, newdata = cvdtest[, -12], type = "response")
pred_train_binary <- ifelse(pred_train > 0.5, 1, 0)
TrainErr <- mean(pred_train_binary != cvdtrain$cardio)
TrainErr

#Naive Bayes Model
mod3 <- naiveBayes(cardio~.,data=cvdtrain)
## Training Error
pred2 <- predict(mod3, newdata=cvdtrain[,-12]);
TrainErr <- c(TrainErr, mean( pred2 != cvdtrain$cardio))
TrainErr 
TestErr <- c(TestErr,  mean( predict(mod3,cvdtest[,-12]) != cvdtest$cardio))
TestErr

Linear Discriminant Analysis
library(MASS)
mod1 <- lda(cardio~.,data=cvdtrain)
pred1 <- predict(mod1, newdata=cvdtrain[,-12])$class
TrainErr <- c(TrainErr,mean(pred1 !=cvdtrain$cardio))
TrainErr
pred1test <- predict(mod1,newdata=cvdtest[,-12])$class
TestErr <- c(TestErr,mean(pred1test != cvdtest$cardio)) 
TestErr

#Cross Validation
set.seed(7406)  # Set seed for randomization
# Initialize the TE values for all models in all B=100 loops
n1 <- dim(cvdtrain)[1]  # Training set sample size
n2 <- dim(cvdtest)[1]   # Testing set sample size
n <- dim(cvd )[1]     # Total sample size
B <- 100         # Number of loops
TEALL <- NULL    # Final TE values
for (b in 1:B) {
  # Randomly select n1 observations as a new training subset in each loop
  flag <- sort(sample(1:n, n1))
  cvdtrain <- cvd [flag,]  # Temp training set for CV
  cvdtest <- cvd [-flag,]  # Temp testing set for CV
  mod1 <- lda(cardio~.,data=cvdtrain)
  pred1test <- predict(mod1,newdata=cvdtest[,-12])$class
  te1 <- mean(pred1test !=cvdtest$cardio) 
  ###Model 2: Naive Bayes
  mod2 <- naiveBayes(cardio~.,data=cvdtrain)
  te2 <-mean( predict(mod2,cvdtest[,-12]) != cvdtest$cardio)
  te2
  ## Method 3:logistic regression) 
  mod3 <- glm(cardio ~ ., family = binomial, data = cvdtrain)
  # Make predictions on the test set
  pred_test <- predict(mod3, newdata = cvdtest[, -12], type = "response")
  # Convert predicted probabilities to binary classes (0 or 1)
  pred_test_binary <- ifelse(pred_test > 0.5, 1, 0)
  # Calculate testing error
  te3 <- mean(pred_test_binary != cvdtest$cardio)
  TEALL <- rbind(TEALL,c(te1,te2,te3))
} 
dim(TEALL); 
colnames(TEALL) <- c("mod1", "mod2", "mod3")
apply(TEALL, 2, mean)

