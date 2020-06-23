### Detecting Credit Card Fraud
### HarvardX Data Science Capstone Project
### Dataset Provider: Machine Learning Group at the Université Libre de Bruxelles (Brussels), hosted on kaggle
### Author: Moritz Christian Schaaf
### GitHub: github.com/mc-schaaf

### ------------------------------------------------------------------------------------------------------------------------
### Prepare Environment
### ------------------------------------------------------------------------------------------------------------------------

# automatically load and install packages
set.seed(42)

# required but not loaded; installed.packages() might take some time on older systems with HDD. 
# If this is the case you can install them manually
tmp <- installed.packages()[,"Package"]
required_but_not_loaded <- c("RColorBrewer", "rafalib", "e1071", "gam", "MASS", "ROSE", "glmnet", "Matrix", "gbm", "leaps", "randomForest")
install_them <- required_but_not_loaded[!required_but_not_loaded %in% tmp]
if(length(install_them) > 0) {install.packages(install_them)}
rm(list = ls())

# required and loaded
suppressMessages(if (!require(tidyverse)) {
  install.packages("tidyverse")
  library(tidyverse)
} else {
  library(tidyverse)
})
suppressMessages(if (!require(caret)) {
  install.packages("caret")
  library(caret)
} else {
  library(caret)
})
suppressMessages(if (!require(data.table)) {
  install.packages("data.table")
  library(data.table)
} else {
  library(data.table)
})


# download data
creditcard <- fread("C:\\Users\\morit\\Downloads\\310_23498_bundle_archive\\creditcard.csv", header = TRUE, sep = ",", dec = ".")
creditcard$Class <- factor(creditcard$Class, labels = c( "0" = "Legitimate", "1" = "Fraud"))






### ------------------------------------------------------------------------------------------------------------------------
### EDA: inspect the data set:
### ------------------------------------------------------------------------------------------------------------------------

# We will inspect the whole dataset to get a first impression of the data. 
# This is performed before the split in test and training data to get an impression if one has to care for some special 
# variables when performing the split


table(creditcard$Class)
# 0 = no fraud, 1 = fraud
# highly imbalanced data


sum(is.na(creditcard))
# no missing data


rafalib::mypar(2,1)
lapply(creditcard[,-c(1,30,31)], var) %>% unlist() %>% plot(xlab = "PC", ylab = "Variance")
abline(h = 1)
title("Eigenvalue of Principal Components")
lapply(creditcard[,-c(1,30,31)], mean) %>% unlist() %>% .[-c(1,30,31)] %>% plot(xlab = "PC", ylab = "mean", ylim = c(-0.1,0.1))
abline(h = 0)
title("Mean of Principal Components")
# PC are centered but not standardized


rafalib::mypar(1,1)
rafalib::imagemat(abs(cor(creditcard[,-c(31)])), col = RColorBrewer::brewer.pal(7,"Greys"))
title("Covariance Matrix of the features")
# there is no covariance between the Principal Components (so rotation happend on the same data the rotation matrix stems from)


creditcard %>% ggplot() + geom_histogram(aes(x = creditcard$Time/(60*60)), color = "black", fill = "grey", bins = 48) +  
  scale_x_continuous("Time (in h)", breaks = 6*c(0:8)) +  ggtitle("Histogram of Time of Transaction") + 
  theme_classic()
creditcard %>% ggplot() + geom_histogram(aes(x = ifelse(Amount < 0.5, 0.5, Amount)), color = "black", fill = "grey", bins = 25) + 
  scale_x_continuous("Amount",trans = "log10") + 
  ggtitle("Histogram of Amount of Transaction", subtitle = paste0("Range: ", range(creditcard$Amount)[1], " up to ", round(range(creditcard$Amount)[2]))) +
  theme_classic()
# there seems to be a circadian pattern in consumer behaviour. A transformation from a 48-hour format to a 24-hour format seems to
# be indicated. In addition, the amount of the transaction is highly scewed and seems to follow a log-normal distribution.
# Here, a transformation seems to be indicated, too. However, small transaction amounts < 50 cents should be rounded towards 0.5 in 
# order to decrese their leverage.
# In practice, one could argue that the loss scales linear to the amount of the fraud and thus large amounts should posses high leverage
# For this report, we will not focus on cost-sensitive training at first.






### ------------------------------------------------------------------------------------------------------------------------
### Split the data 
### ------------------------------------------------------------------------------------------------------------------------

# In this scenario we will only use use one training and one test set and tune the algorithms by cross-validation or bootstrapping.
# The reason for doing this is the low prevalence of fraudulent transactions. However, keep in mind that the performance on the test set
# is a random process and the true performance of the "best" algorithm may be a statistical artifact. For further information concerning this topic
# please look up "regression to the mean"


# Test set will be 33% of the data
suppressWarnings(set.seed(1, sample.kind="Rounding"))
test_index <- createDataPartition(y = creditcard$Class, times = 1, p = 1/3, list = FALSE)
data_test <- creditcard[test_index,] %>% as.data.frame()
data_train <- creditcard[-test_index,] %>% as.data.frame()
rm(test_index)
rm(creditcard)

# Split Test 50-50 again for final evaluation
test_index <- createDataPartition(y = data_test$Class, times = 1, p = 1/2, list = FALSE)
data_validation <- data_test[test_index,]
data_test <- data_test[-test_index,]
rm(test_index)


# EDA2: Inspect the relevant variables in detail
# depending on the performance of your PC this might take some time!
do_plot_scatter <- function(i) {
  data_train %>% ggplot(aes(x = data_train[,i], y = data_train[,i+1])) + 
    geom_point(aes(color = Class), alpha = 0.1) + 
    geom_point(aes(color = Class), alpha = ifelse(data_train$Class == 1, 0.5, 0)) + 
    xlab(paste0(colnames(data_train)[i])) +
    ylab(paste0(colnames(data_train)[i+1])) +
    theme_bw() +
    scale_color_manual(name = "Fraudulent?", values = c("black", "red"), labels = c("Legitimate" = "No", "Fraud" = "Yes"))
}

if (1 == 2){
  sapply(seq(2, 28, by = 2), function(x) {plot(do_plot_scatter(x))})
}

# some PCs do not seem normally distributed
# In addition, there seems to be no multivariate normality 
# However, on some PCs there seems to be a seperation between the fraudulent and normal transactions


do_plot_distribution <- function(i) {
  limits0 <- data_train[data_train$Class == 0, i] %>% quantile(c(0.25, 0.975))
  limits1 <- data_train[data_train$Class == 1, i] %>% quantile(c(0.25, 0.975))
  limits <- c(min(c(limits0, limits1)), max(c(limits0, limits1)))
  data_train %>% ggplot(aes(x = data_train[,i], color = Class, fill = Class)) + 
    geom_density(bw = 0.5, alpha = 0.2) +
    scale_x_continuous(paste0(colnames(data_train)[i])) +
    scale_y_continuous("Density", labels = NULL) +
    theme_bw() +
    scale_color_manual(name = "Fraudulent?", values = c("black", "red"), labels = c("Legitimate" = "No", "Fraud" = "Yes")) +
    scale_fill_manual(name = "Fraudulent?", values = c("black", "red"), labels = c("Legitimate" = "No", "Fraud" = "Yes")) +
    coord_cartesian(xlim = limits)
}
if(1 == 2){
  sapply(2:(ncol(data_train)-2), function(x) {plot(do_plot_distribution(x))})
}
# However, on some PCs there seems to be a seperation between the fraudulent and normal transactions
# The Principal Components do not need any feature engineering. Outliers seem to be mainly driven by fraudulent transactions
# thus they should not be removed/shrunken


# due to heavy skew on log10 scale
data_train %>% ggplot(aes(x = Amount+1, color = Class, fill = Class)) + 
  geom_density(bw = 0.5, alpha = 0.2) +
  scale_x_continuous("Amount", trans = "log10") +
  scale_y_continuous("Density", labels = NULL) +
  theme_bw() +
  scale_color_manual(name = "Fraudulent?", values = c("black", "red"), labels = c("Legitimate" = "No", "Fraud" = "Yes")) +
  scale_fill_manual(name = "Fraudulent?", values = c("black", "red"), labels = c("Legitimate" = "No", "Fraud" = "Yes"))


# due to two-day intervall reduced to 24 hours
data_train %>% 
  ggplot(aes(x = ifelse(Time > 24*60*60, Time - 24*60*60, Time) / (60*60), 
             color = Class, fill = Class)) + 
  geom_density(bw = 2, alpha = 0.2) +
  scale_x_continuous("Time of Day (in h)", breaks = 6*c(0:4)) +
  scale_y_continuous("Density", labels = NULL) +
  theme_bw() +
  scale_color_manual(name = "Fraudulent?", values = c("black", "red"), labels = c("Legitimate" = "No", "Fraud" = "Yes")) +
  scale_fill_manual(name = "Fraudulent?", values = c("black", "red"), labels = c("Legitimate" = "No", "Fraud" = "Yes"))






### ------------------------------------------------------------------------------------------------------------------------
### Feature Engineering
### ------------------------------------------------------------------------------------------------------------------------

# As we have mostly principal components, a lot of the feature engineering has already been done for us.
# However, we will change the Amount to log10 scale and the Time to Time of Day
data_train <- data_train %>% mutate(
  Time = ifelse(Time > 24*60*60, Time - 24*60*60, Time),
  Amount = log10(Amount+1)
)
data_test <- data_test %>% mutate(
  Time = ifelse(Time > 24*60*60, Time - 24*60*60, Time),
  Amount = log10(Amount+1)
)
data_validation <- data_validation %>% mutate(
  Time = ifelse(Time > 24*60*60, Time - 24*60*60, Time),
  Amount = log10(Amount+1)
)





### ------------------------------------------------------------------------------------------------------------------------
### Baseline Measurements
### ------------------------------------------------------------------------------------------------------------------------

# The most simple approach is always predicting to the a priori.
predictions <- data.frame(guessing = sample(unique(data_train[,31]), prob = c(1, 0), size =  nrow(data_test), replace = TRUE))

caret::confusionMatrix(predictions$guessing, data_test$Class, positive = "Fraud")$table
caret::confusionMatrix(predictions$guessing, data_test$Class, positive = "Fraud")$byClass
rm(predictions)
# We see that Accuracy does not make much sense here as it is 99% with random guessing alone.
# We will thus focus on other performance metrices as well
# Lets see how these two behave when not guessing towards the a priori but randomly with probability p


suppressWarnings(set.seed(42, sample.kind = "Rounding"))
ps <- seq(0,1, by = 0.01)
guessing <- map_df(ps, function(p){
  y_hat <- sample(unique(data_train[,31]), prob = c(p, 1-p), size =  nrow(data_test), replace = TRUE)
  SS <- caret::confusionMatrix(y_hat, data_test$Class, positive = "Fraud")$byClass[c("Sensitivity", "Specificity", "F1")]
  AC <- caret::confusionMatrix(y_hat, data_test$Class, positive = "Fraud")$overall["Accuracy"]
  list(method = "guessing", cutoff = p, sensitivity = SS[1], specificity = SS[2], F1 = SS[3], accuracy = AC[1])
})
guessing %>% ggplot(aes(x = cutoff)) + 
  geom_line(aes(y = sensitivity, color = "True Positive Rate")) + 
  geom_line(aes(y = 1 - specificity, color = "False Positive Rate")) + 
  geom_line(aes(y = specificity, color = "Specificity")) + 
  geom_line(aes(y = accuracy, color = "Accuracy")) +
  geom_line(aes(y = F1, color = "F1")) + 
  theme_bw() +
  scale_color_discrete("Error Rate Type") + 
  ggtitle("Compairson of different evaluation metrices on this problem")
# We see that Accuracy and Specificity are almost overlapping. This is due to the high bias in the prevalences.
# The false positive rate is defined as 1 - specificity. Therefore, Accuracy, Specificity and FPR als have the same information
# in this scenario
# The F1 Score is extremely low du to the low values of precision and recall (also due to the prevalence)
# This calls for the use of a ROC curve for evaluating this problem:


# define a function that draws a ROC curve
do_ROC <- function(dataframe){ 
  dataframe %>% ggplot(aes(x = 1 - specificity, y = sensitivity, color = method)) + 
  geom_line() + 
  geom_abline(intercept = 0, slope = 1) + 
  theme_bw() +
  scale_color_discrete("Method") + 
  xlab("1 - specificity\n = False Positive Rate\n ~ 1 - Accuracy in this case") + 
  ggtitle("ROC curve for different prediction methods")
}


# plot the guessing approach
do_ROC(guessing)
# We see that the guessing approach is very similar to the identity line (or "no information line")


# The most simple machine learning algorithm is logistic regression.
# It is highly interpretable, thus:
# "If you can't beat it, stick with it"

suppressWarnings(set.seed(42, sample.kind = "Rounding"))
fit_logistic <- glm(Class ~ ., data = data_train, family = "binomial")
y_hat_p <- predict(fit_logistic, newdata = data_test, type = "response")

# We will write a function that automatically evaluates the algorithm's probability esimations with different cutoff values
do_eval <- function(data, method_name = "please specify", ps = seq(0,1, by = 0.001)){
  output <- map_df(ps, function(p){
    y_hat <- ifelse(data < p, "Legitimate", "Fraud") %>% factor(levels = levels(data_train$Class))
    SS <- caret::confusionMatrix(y_hat, data_test$Class, positive = "Fraud")$byClass[c("Sensitivity", "Specificity", "F1")]
    AC <- caret::confusionMatrix(y_hat, data_test$Class, positive = "Fraud")$overall["Accuracy"]
    list(method = method_name, cutoff = p, sensitivity = SS[1], specificity = SS[2], F1 = SS[3], accuracy = AC[1])
  })
  return(output)
}


# apply and plot ROC 
ROC_all <- rbind(guessing, do_eval(data = y_hat_p, method_name = "logistic"))
do_ROC(ROC_all)
rm(guessing)






### ------------------------------------------------------------------------------------------------------------------------
### Advanced Machine Learning Techniques 1: Linear Model Selection and Motivation for Sampling Techniques
### ------------------------------------------------------------------------------------------------------------------------

# We most likely overfit our logistic regression and introduced noise into the data. 
# Hence, we will try different approaches to shrink the parameters (also called Regularization or Constraining the Coefficients)
# We will try and discuss four different approaches: Feature Selection, Ridge-Regression, Lasso-Regression and PC-Regression


# The first step is trying out feature selection. We fit different regressions and try leaving out different predictors (only 1 at a time). 
# The best model with n parameters is selected. We define this model as new base model and define n <- n - 1.
# We iterate through this throw-out process until some criterion (here AIC) has reached a (possibly local) minimum
suppressWarnings(set.seed(42, sample.kind = "Rounding"))
if (1 == 2) {
  logistic_backwards <- MASS::stepAIC(fit_logistic, direction = "backward")
}
# final model: 
# Step:  AIC=1569.36
# Class ~ V1 + V4 + V6 + V8 + V9 + V10 + V12 + V13 + V14 + V15 + 
#   V16 + V20 + V21 + V22 + V23 + V27 + V28 + Amount
# this fits sum(32:19) = 357 models and thus takes an eternity with 200 000 observations

# apply the regularized model
suppressWarnings(set.seed(42, sample.kind = "Rounding"))
fit_logistic <- glm(Class ~ V1 + V4 + V6 + V8 + V9 + V10 + V12 + V13 + V14 + 
                      V15 + V16 + V20 + V21 + V22 + V23 + V27 + V28 + Amount, 
                    data = data_train, family = "binomial")
y_hat_p <- predict(fit_logistic, newdata = data_test, type = "response")

ROC_all <- rbind(ROC_all, do_eval(data = y_hat_p, method_name = "logistic backwards"))
do_ROC(ROC_all)

# We will use a method that tackles this computing time problem as well as the problem of the low prevalence and 
# unusable Accuracy measures
# We will under-sample the class with the higher prevalence
# for this we will use the ROSE package

suppressWarnings(set.seed(42, sample.kind = "Rounding"))
data_train_under <- ROSE::ovun.sample(Class ~ ., data_train, method = "under", p = 1/3)$data

suppressWarnings(set.seed(42, sample.kind = "Rounding"))
fit_logistic <- glm(Class ~ V1 + V4 + V6 + V8 + V9 + V10 + V12 + V13 + V14 + 
                      V15 + V16 + V20 + V21 + V22 + V23 + V27 + V28 + Amount, 
                    data = data_train_under, family = "binomial")
y_hat_p <- predict(fit_logistic, newdata = data_test, type = "response")

ROC_all <- rbind(ROC_all, do_eval(data = y_hat_p, method_name = "logistic undersampling"))
do_ROC(ROC_all)

fit_logistic <- MASS::stepAIC(fit_logistic, direction = "backward")
y_hat_p <- predict(fit_logistic, newdata = data_test, type = "response")

ROC_all <- rbind(ROC_all, do_eval(data = y_hat_p, method_name = "logistic undersampling\n + backwards"))
do_ROC(ROC_all)
# loosing a massive amount of information (99.5% of the training data!!!) is not too bad. It actually helps the process as it makes
# some algorithms computationally feasible



### ------------------------------------------------------------------------------------------------------------------------
### Advanced Machine Learning Techniques 2: PC-Regression, Ridge and Lasso
### ------------------------------------------------------------------------------------------------------------------------

# Moving forward with the reduced training set we will apply further shrinkage and regularization methods
# The first one is Principal Components Regression. One computes all n PCs on the predictors and uses a selection m < n of those PCs to 
# predict. The advantage over stepwise regression is the number of models fitted. As the PCs are ordered, only n+1 models have to be
# fit instead of sum(1:(n+1)) models. I do not know of any packages that perform this kind of selection on logistic regressions,
# so we will write our own function. This is by no means an elegant function, but it does the trick and should show you the general idea.
# We will use the AIC as an indicator for model quality. This is ok as we have the 33-67 prevalence split

# Baseline Model contains Time and Amount
X <- data_train_under %>% select(Time, Amount, Class)
X_holdout <- data_train_under %>% select(-Time, -Amount, -Class)
PC_eval <- map_df(1:ncol(X_holdout), function(i){
  # in each step 1 PC is added on top of the already existing ones
  X_tmp <- cbind(X, X_holdout %>% select(1:i))
  fit_tmp <- glm(Class ~ . , data = X_tmp, family = "binomial")
  list(call = paste0(fit_tmp$terms)[3], AIC = fit_tmp$aic)
})
best_call <- PC_eval %>% top_n(1, -AIC) %>% pull(call)

# use the best call and fit this model
suppressWarnings(set.seed(42, sample.kind = "Rounding"))
fit_logistic <- glm(as.formula(paste0("Class ~ ",best_call)), 
                    data = data_train_under, family = "binomial")

y_hat_p <- predict(fit_logistic, newdata = data_test, type = "response")
ROC_all <- rbind(ROC_all, do_eval(data = y_hat_p, method_name = "logistic PCR"))
do_ROC(ROC_all)

# up until now we selected some features and used the "whole" coefficients for predictions
# now we will take all features but regularize/ shrink some of the coefficients.
# In the course we saw a quick example of regulating with the L2-Norm (Ridge-Regression)
# However, one could also use the L1-Norm (Lasso) or a combination of both (elastic net)
# the difference is in the weight of the coeffifients: L2 = sum(betas^2); L1 = sum(abs(betas))
# therefore we will use the glmnet package 

# This package needs a special format
# We will leave out the Time and Amount column and use only the PCs
X <- model.matrix(Class~., data_train_under %>% select(-Time, -Amount))
Y <- ifelse(data_train_under$Class == "Fraud", 1, 0)

# determine optimal shrinkage by cross-validation
rafalib::mypar(2,1)
suppressWarnings(set.seed(42, sample.kind = "Rounding"))
fit_ridge <- glmnet::glmnet(X, Y, alpha = 0, family = "binomial")
cv_ridge <- glmnet::cv.glmnet(X, Y, alpha = 0, family = "binomial")
plot(cv_ridge, xlab = "Log Lambda")
plot(fit_ridge, xvar = "lambda")
abline(v = log(cv_ridge$lambda.min), lty = 3)
abline(v = log(cv_ridge$lambda.1se), lty = 3)

suppressWarnings(set.seed(42, sample.kind = "Rounding"))
fit_lasso <- glmnet::glmnet(X, Y, alpha = 1, family = "binomial")
cv_lasso <- glmnet::cv.glmnet(X, Y, alpha = 1, family = "binomial")
plot(cv_lasso, xlab = "Log Lambda")
plot(fit_lasso, xvar = "lambda")
abline(v = log(cv_lasso$lambda.min), lty = 3)
abline(v = log(cv_lasso$lambda.1se), lty = 3)
rafalib::mypar(1,1)
# The L1 Norm (Lasso) seems to be more effective in reducing the Binomial Deviance. 
# In addition it creates a sparse predictor matrix, so it "removes" some predictors by setting their coefficients to zero:
fit_lasso <- glmnet::glmnet(X, Y, alpha = 1, family = "binomial", lambda = cv_lasso$lambda.1se)
fit_lasso$beta %>% plot(xlab = "Coefficient Index", ylab = "Coefficient Value", 
                         col = ifelse(fit_lasso$beta != 0, "black", "lightgrey"), type = "b", pch = 19)


# How is the performance of this sparse predictor matrix?
X_new <- model.matrix(Class~., data_test %>% select(-Time, -Amount))
y_hat_p <- predict(fit_lasso, newx = X_new, type = "response")

ROC_all <- rbind(ROC_all, do_eval(data = y_hat_p, method_name = "logistic lasso"))
do_ROC(ROC_all)
# even when using only roughly 0.5% of the observations and 40% of the predictors, the lasso manages to outperform the 
# plain vanilla logistic regression
rm(list = c("cv_lasso", "cv_ridge", "fit_lasso", "fit_logistic", "fit_ridge", "X", "X_new", "Y"))






### ------------------------------------------------------------------------------------------------------------------------
### Advanced Machine Learning Techniques 3: kNN and bagged/ boosted Regression Trees
### ------------------------------------------------------------------------------------------------------------------------

# Up until now all predictions were made using adding up of different predictions (= coefficients)
# This makes for highly interpretable results.
# However, often there are nonlinear and interaction effects that can not be captured by linear models
# One could use support vector machines which compute in n-dimensional space
# However, the idea behind SVMs is too closely related to GLMs/GAMs
# Some statistics researchers claim that kNN is the optimal method for 1/3 of machine learning methods
# Others claim the same for Decision Trees in different versions (random forests, bagged, boosted, etc)
# We will therefore look at kNN, RF, and boosting methods and compaire them to elastic net and plain vanilla logistic regression
# We will utilize the caret package, as it allows us to write in coherent syntax

# We will use some fast 10-fold CV to get first impressions of the algorithm performance
control_parameters <- caret::trainControl(method = "repeatedcv", number = 10, repeats = 10)


# We will set up some tuning parameter grids 
grid_plainvanilla_reg <- expand.grid(alpha = 0, lambda = 0) # this equals regular regression
grid_glmnet <- expand.grid(alpha = seq(0, 1, by = 1/3), 
                           lambda = exp(seq(-10, -3, by = 0.5)))
grid_knn <- expand.grid(k = seq(3, 51, by = 2))
grid_GBM <- expand.grid(n.trees = c(5000),             # those values were selected directly with the gbm package
                        interaction.depth = c(5),      # as this takes very long computationally with caret
                        shrinkage = c(0.001),          # You don't have to cv as you can use the oob esimate for the performance
                        n.minobsinnode = c(17))        # feel free to also try different other values yourself
grid_RF <- expand.grid(mtry = c(1, 3, 5, 7))

# Do Plain Vanilla Logistic Regression with caret package
suppressWarnings(set.seed(999, sample.kind = "Rounding"))
fit_plainvanilla_reg <- caret::train(Class ~., data = data_train_under, 
                                     method = "glmnet", tuneGrid = grid_plainvanilla_reg,
                                     preProcess = c("scale", "center"), trControl = control_parameters)
p_Fraud <- predict(fit_plainvanilla_reg, newdata = data_test, type = "prob")[,2]

ROC_caret <- do_eval(data = p_Fraud, method_name = "Plain Vanilla\nLogistic Regression")
do_ROC(ROC_caret)


# Do Elastic Net Logistic Regression with caret package
suppressWarnings(set.seed(999, sample.kind = "Rounding"))
fit_glmnet <- caret::train(Class ~., data = data_train_under, 
                                     method = "glmnet", tuneGrid = grid_glmnet,
                                     preProcess = c("scale", "center"), trControl = control_parameters)
p_Fraud <- predict(fit_glmnet, newdata = data_test, type = "prob")[,2]

ggplot(fit_glmnet, highlight = TRUE) + theme_classic() + scale_x_continuous(trans = "log10")
fit_glmnet$bestTune

ROC_caret <- rbind(ROC_caret, do_eval(data = p_Fraud, method_name = "Elastic Net\n Regression"))
do_ROC(ROC_caret)


# Do kNN with caret package
suppressWarnings(set.seed(999, sample.kind = "Rounding"))
fit_knn <- caret::train(Class ~., data = data_train_under, 
                           method = "knn", tuneGrid = grid_knn,
                           preProcess = c("scale", "center"), trControl = control_parameters)
p_Fraud <- predict(fit_knn, newdata = data_test, type = "prob")[,2]

ggplot(fit_knn, highlight = TRUE) + theme_classic() #+ scale_x_continuous(trans = "log10")
fit_knn$bestTune

ROC_caret <- rbind(ROC_caret, do_eval(data = p_Fraud, method_name = "kNN"))
do_ROC(ROC_caret)


# Do Random Forest with caret package
suppressWarnings(set.seed(999, sample.kind = "Rounding"))
fit_RF <- caret::train(Class ~., data = data_train_under, 
                        method = "rf", tuneGrid = grid_RF, ntree = 1000,
                        preProcess = c("scale", "center"), trControl = control_parameters)
p_Fraud <- predict(fit_RF, newdata = data_test, type = "prob")[,2]

ggplot(fit_RF, highlight = TRUE) + theme_classic()

ROC_caret <- rbind(ROC_caret, do_eval(data = p_Fraud, method_name = "RF"))
do_ROC(ROC_caret)


# Do Boosting with caret package
suppressWarnings(set.seed(999, sample.kind = "Rounding"))
fit_gbm <- caret::train(Class ~., data = data_train_under, 
                           method = "gbm", verbose = FALSE, distribution = "bernoulli", tuneGrid = grid_GBM,
                           preProcess = c("scale", "center"), trControl = control_parameters)
p_Fraud <- predict(fit_gbm, newdata = data_test, type = "prob")[,2]

ROC_caret <- rbind(ROC_caret, do_eval(data = p_Fraud, method_name = "Boosted Tree"))
do_ROC(ROC_caret)






### ------------------------------------------------------------------------------------------------------------------------
### Which Model to Take
### ------------------------------------------------------------------------------------------------------------------------
stop("Work under Progress")
do_ROC(ROC_caret) + coord_cartesian(xlim = c(0, 0.005))










### ------------------------------------------------------------------------------------------------------------------------
### Outdated Code Snippets
### ------------------------------------------------------------------------------------------------------------------------

### testing GBM
data_train_under2 <- data_train_under %>% mutate(Class = as.numeric(Class)-1)

rafalib::mypar(2,4)
for (i in c(5, 7)) {
  for (j in c(13, 17, 21, 25) ){
    tmp_gbm <- gbm::gbm(Class ~ ., distribution = "bernoulli", data = data_train_under2,
                        n.trees = 5/0.001,
                        interaction.depth = i,
                        n.minobsinnode = j,
                        shrinkage = 0.001,
                        bag.fraction = 0.5,
                        train.fraction = 0.75)
    gbm::gbm.perf(tmp_gbm, method = "test")
    title(paste0("Treedepth = ", i, ", MinObs = ", j, ",\nBest =", round(min(tmp_gbm$valid.error),3)))
  }
}



### ------------------------------------------------------------------------------------------------------------------------
### To Do
### ------------------------------------------------------------------------------------------------------------------------

# automatically download data