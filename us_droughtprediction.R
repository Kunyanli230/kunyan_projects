setwd("C:/Users/lihea/OneDrive/Desktop/STAT 362/")
data <- read.csv("train_timeseries.csv")
soil_data <- read.csv("soil_data.csv")
data <- read.csv("validation_timeseries.csv")
head(data)
colnames(data)
library(ROCR)
library(DMwR)
library(ggplot2) 
library(maps)
library(tidyverse)
library(ISLR2)
library(randomForest)
library(tree)
library(caret)
library(psych)
colnames(data)
data = na.omit(data)
soil_data = na.omit(soil_data)

data_long <- data %>%
  select(T2M, T2MDEW, T2MWET ) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")

# Plot the combined distribution
ggplot(data_long, aes(x = Value, fill = Variable)) +
  geom_histogram(position = "identity", alpha = 0.6, binwidth = 0.5) +
  labs(title = "Combined Distribution of T2M, T2MDEW, T2MWET ",
       x = "Value",
       y = "Frequency") +
  scale_fill_manual(values = c("T2M" = "blue", "T2MDEW" = "red", "T2MWET" = "green"))


train_data = data[train_idx, !(colnames(data) %in% 'date')]

cor_matrix <- cor(data[colnames(data)])
cor_matrix

cor_matrix_all <- cor(train_data, use = "complete.obs")

# Melt the correlation matrix for plotting
cor_melted_all <- melt(cor_matrix_all)

# Plot the heatmap
ggplot(cor_melted_all, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1,1)) +
  labs(title = "Correlation Matrix Heatmap for All Variables",
       x = "",
       y = "") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 3)

# Amount of data to condense to
AMOUNT_DATA = 10000
UNUSED_COLS = c("T2MWET", "T2MDEW", "WS10M_MAX", "WS10M_MIN", "WS10M_RANGE")
set.seed(11)

# Condensing dataset
data_idx = sample(1:nrow(data), AMOUNT_DATA, replace = FALSE)
data = data[data_idx,]

# Splitting up training and testing data
train_idx = sample(1:nrow(data), 0.9 * AMOUNT_DATA, replace=F)
train_data = data[train_idx, !(colnames(data) %in% UNUSED_COLS)]
test_data = data[-train_idx, !(colnames(data) %in% UNUSED_COLS)]
train_data <- train_data %>%
  mutate(date = as.numeric(date) - as.numeric(as.Date("2000-1-1")), score = round(score)) %>%
  mutate(season = cospi(2 * (date - 172) / 365)) %>%
  left_join(soil_data, by = "fips")
test_data <- test_data %>%
  mutate(date = as.numeric(date) - as.numeric(as.Date("2000-1-1")), score = round(score)) %>%
  mutate(season = cospi(2 * (date - 172) / 365)) %>%
  left_join(soil_data, by = "fips")
# Separate features (X_train) and target variable (y_train)
X_train <- train_data[, -ncol(train_data)]
y_train <- train_data[, ncol(train_data)]

# Apply Neighbourhood Cleaning Rule for under-sampling
undersampled_data <- ubNCR(X_train, y_train, k = 3, cleanTh = 0.5)
# Make predictions and evaluate the model
# (Replace 'your_prediction_function' with the appropriate function for your model)
train_pred <- your_prediction_function(fit, train_data)
test_pred <- your_prediction_function(fit, test_data)

# Splitting up training and testing data
train_idx = sample(1:nrow(data), 0.9 * AMOUNT_DATA, replace=F)
train_data = data[train_idx, !(colnames(data) %in% UNUSED_COLS)]
test_data = data[-train_idx, !(colnames(data) %in% UNUSED_COLS)]


rf_fit <- randomForest(score ~., data = train_data, 
                       mtry = (ncol(train_data) - 1)/ 3, 
                       ntree = 50, 
                       importance=T)

train_pred = round(predict(rf_fit, train_data))
test_pred = round(predict(rf_fit, test_data))

train_response = round(train_data$score)
test_response = round(test_data$score)

# Train accuracy
sum(train_pred == train_response) / length(train_response)

# Test accuracy
sum(test_pred == test_response) / length(test_response)

# Train confusion
table(train_pred, train_response)

# Test confusion
table(test_pred, test_response)
importance(rf_fit)
varImpPlot(rf_fit)
most_important_var <- rownames(importance(rf_fit))[which.max(importance(rf_fit)[, 1])]
most_important_var

# Set the number of bootstrap samples
bootstrap_size <- floor(0.8 * nrow(train_data))

# Fit the random forest model with bootstrapping
rf_fit <- randomForest(score ~., data = train_data, 
                       mtry = (ncol(train_data) - 1) / 3, 
                       ntree = 50, 
                       importance = TRUE,
                       sampsize = bootstrap_size)

# Make predictions on the train and test data
train_pred <- round(predict(rf_fit, train_data))
test_pred <- round(predict(rf_fit, test_data))

# Calculate the actual responses
train_response <- round(train_data$score)
test_response <- round(test_data$score)

# Calculate and print the train accuracy
train_accuracy <- sum(train_pred == train_response) / length(train_response)
cat("Train accuracy:", train_accuracy, "\n")

# Calculate and print the test accuracy
rf_accuracy_boot <- sum(test_pred == test_response) / length(test_response)
cat("Test accuracy:", rf_accuracy_boot, "\n")

# Print the train confusion matrix
cat("Train confusion matrix:\n")
print(table(train_pred, train_response))

# Print the test confusion matrix
cat("Test confusion matrix:\n")
print(table(test_pred, test_response))
library(nnet)
fit <- multinom(score ~ ., data = train_data)

train_pred = predict(fit, train_data)
test_pred = predict(fit, test_data)

train_response = round(train_data$score)
test_response = round(test_data$score)

# Train accuracy
mean(train_pred == train_response)

# Test accuracy
mean(test_pred == test_response)

# Train confusion
table(train_pred, train_response)

# Test confusion
table(test_pred, test_response)


# LASSO Regression
X <- as.matrix(train_data[, c(-1, -2, -3, -17)])
y <- as.matrix(train_data$score)
fit <- cv.glmnet(X, y, alpha = 1)
X_test <- as.matrix(test_data[, c(-1, -2, -3, -17)])
y_test <- as.matrix(test_data$score)
train_pred <- predict(fit, X)
test_pred <- predict(fit, X_test)

# Calculate accuracies
train_accuracy <- mean(round(train_pred) == round(train_data$score))
test_accuracy <- mean(round(test_pred) == round(test_data$score))
test_accuracy
la_accuracy <- test_accuracy

# ROC
library(pROC)
X <- as.matrix(train_data[, c(-1, -2, -3, -17)])
y <- as.matrix(train_data$score)
fit <- cv.glmnet(X, y, alpha = 1)
X_test <- as.matrix(test_data[, c(-1, -2, -3, -17)])
y_test <- as.matrix(test_data$score)
train_pred <- predict(fit, X)
test_pred <- predict(fit, X_test)

# Calculate accuracies
train_accuracy <- mean(round(train_pred) == round(train_data$score))
test_accuracy <- mean(round(test_pred) == round(test_data$score))

# ROC curve
roc_curve <- roc(response = y_test, predictor = as.numeric(test_pred))
plot(roc_curve, main = "ROC Curve")
# knn test
# remove non-numeric columns (fips and date)
cols <- c("fips", "date")
train <- train[, !(colnames(train) %in% cols)]
test <- test[, !(colnames(test) %in% cols)]

row.names(train) <- train_indices[1:nrow(train)]
row.names(test) <- test_indices[1:nrow(test)]

# need to modify the score to become discrete rather than continuous 
train$score <- round(train$score)
test$score <- round(test$score)


# labels (we are categorizing based on drought score)
train_labels <- train$score
test_labels <- test$score


# Normalize the data using min max
# Z-normalization gives almost identical values so it made no difference
train_n <- train
test_n <- test

train_min <- apply(train, 2, min, na.rm = TRUE)
train_max <- apply(train, 2, max, na.rm = TRUE)


for(i in 1:ncol(train)){
  train_n[,i] <- (train[,i] - train_min[i]) / (train_max[i] - train_min[i])
  test_n[,i] <- (test[,i] - train_min[i]) / (train_max[i] - train_min[i])
}

# now run the KNN classification
library(class)
knn_predict <- knn(train = train_n, test = test_n, cl = train_labels, k = 5)

# evaluating the results 
confusion_matrix <- table(test_labels, knn_predict)
confusion_matrix
#               knn_predict
#test_labels   0   1   2   3   4   5
#          0 329   1   0   0   0   0
#          1  17 132   1   0   0   0
#          2   0  14  82   2   0   0
#          3   0   0   6  56   1   0
#          4   0   0   0   8  14   5
#          5   0   0   0   0   4  42

accuracy <- sum(diag(confusion_matrix)) / length(test_labels) * 100
accuracy # 90.6

train_data_smote <- SMOTE(Class ~ ., data = train_data, k = 5, perc.over = 100, perc.under = 200)

# Train a KNN model on the balanced data
library(class)
train_data_smote <- SMOTE(Class ~ ., data = train_data, k = 5, perc.over = 100, perc.under = 200)

knn_model <- knn(train = train_data_smote[,-which(names(train_data_smote) == "Class")],
                 test = test_data[,-which(names(test_data) == "Class")],
                 cl = train_data_smote$Class,
                 k = 5)

# Define the number of bootstraps
n_bootstraps <- 100

# Create a list to store the predictions for each bootstrap
bootstrap_preds <- list()

# Create a list to store the models for each bootstrap
bootstrap_models <- list()

# Bootstrap loop
for (i in 1:n_bootstraps) {
  # Create a bootstrap sample of the training data
  boot_indices <- sample(1:nrow(train_data), replace = TRUE)
  X_boot <- as.matrix(train_data[boot_indices, c(3:15, 17:48)])
  y_boot <- as.matrix(train_data$score[boot_indices])
  
  # Train a model on the bootstrap sample
  fit_boot <- cv.glmnet(X_boot, y_boot, alpha = 1)
  bootstrap_models[[i]] <- fit_boot
  
  # Make predictions on the test data
  test_pred_boot <- predict(fit_boot, X_test)
  bootstrap_preds[[i]] <- test_pred_boot
}

# Aggregate the predictions from each bootstrap model
# You can use different aggregation methods such as mean, median, or majority vote
final_test_pred <- rowMeans(do.call(cbind, bootstrap_preds))

# Calculate the final accuracy
final_test_accuracy <- mean(round(final_test_pred) == round(test_data$score))
final_test_accuracy
knn_accuracy_boot <- bootstrap_knn_accuracy

rf_accuracy
mn_accuracy
knn_accuracy
la_accuracy
# Create a data frame for ANOVA
performance_data <- data.frame(
  Model = rep(c("RandomForest", "Multinomial", "KNN","LASSO"), each = 1),
  Accuracy = c(rf_accuracy, mn_accuracy, knn_accuracy,la_accuracy)
)

# Perform ANOVA
anova_result <- aov(Accuracy ~ Model, data = performance_data)
summary(anova_result)

ggplot(performance_data, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Model Performance Comparison",
       x = "Model",
       y = "Accuracy") 

rf_accuracy_boot
mn_accuracy_boot
knn_accuracy_boot
la_accuracy_boot
performance_data <- data.frame(
  Model = rep(c("RandomForestCV", "MultinomialCV", "KNNCV","LASSOCV"), each = 1),
  Accuracy = c(rf_accuracy_boot, mn_accuracy_boot, knn_accuracy_boot,la_accuracy_boot)
)

# Perform ANOVA
anova_result <- aov(Accuracy ~ Model, data = performance_data)
summary(anova_result)

ggplot(performance_data, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Model Performance Comparison",
       x = "Model",
       y = "Accuracy") 
