# Load necessary libraries
library(kknn)
library(e1071)
library(caret)
library(ggplot2)

# Load the data-set
data <- read.table("../datasets/ISY6501_HW1_credit_card_data-headers.txt", header = TRUE)

# Separate predictors and response
predictors <- data[, -ncol(data)]
response <- as.factor(data[, ncol(data)])

# Handle missing values (drop rows with NA)
data <- na.omit(data)

# Scale continuous variables
continuous_columns <- c("A2", "A3", "A8", "A11", "A14", "A15")
data[continuous_columns] <- scale(data[continuous_columns])

# Set Up Predictors (Independent Values) and Response (Dependent Value) and size N
predictors <- as.matrix(data[, 1:10])
response <- as.factor(data[, 11])
n <- nrow(data)






#------------------------------------------------------------------#






# Part (a): k-NN with Cross-Validation
set.seed(123)  # For reproducibility
cv_control <- trainControl(method = "cv", number = 10)

grid <- expand.grid(kmax = seq(1, 15, by = 2), distance = 2, kernel = "optimal")  # Tuning grid for k values

knn_model <- train(
  x = predictors,
  y = response,
  method = "kknn",
  tuneGrid = grid,
  trControl = cv_control
)

print("Best k from cross-validation:")
print(knn_model$bestTune)

# Plot cross-validation results
cv_results <- as.data.frame(knn_model$results)
ggplot(cv_results, aes(x = kmax, y = Accuracy)) +
  geom_line() +
  geom_point() +
  labs(title = "Cross-Validation Accuracy vs. k", x = "Number of Neighbors (k)", y = "Accuracy") +
  theme_minimal()






#------------------------------------------------------------------#






# Part (b): Train/Validation/Test Split
set.seed(123)
train_idx <- createDataPartition(response, p = 0.6, list = FALSE)
train_data <- data[train_idx, ]
remaining_data <- data[-train_idx, ]

# Split remaining into validation and test
val_idx <- createDataPartition(remaining_data[, ncol(remaining_data)], p = 0.5, list = FALSE)
val_data <- remaining_data[val_idx, ]
test_data <- remaining_data[-val_idx, ]

# Train k-NN on training data with best k from cross-validation
best_k <- knn_model$bestTune$kmax

knn_final <- kknn(
  formula = as.formula(paste0(names(train_data)[ncol(train_data)], " ~ .")),
  train = train_data,
  test = val_data,
  k = best_k,
  distance = 2,
  kernel = "optimal"
)

# Evaluate on validation data
val_predictions <- fitted(knn_final)
# Ensure factor levels match
val_predictions <- factor(val_predictions, levels = levels(as.factor(val_data[, ncol(val_data)])))
conf_matrix <- confusionMatrix(val_predictions, as.factor(val_data[, ncol(val_data)]))

print("Validation Performance:")
print(conf_matrix)

# Visualize the confusion matrix for validation data
conf_matrix_df <- as.data.frame(conf_matrix$table)
ggplot(conf_matrix_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  labs(title = "Confusion Matrix: Validation Data", x = "Actual", y = "Predicted") +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_minimal()





#------------------------------------------------------------#




# OPTIONAL SVM Training
# Train SVM (optional) on training data
svm_model <- svm(
  x = as.matrix(train_data[, -ncol(train_data)]),
  y = as.factor(train_data[, ncol(train_data)]),
  kernel = "radial",
  cost = 1
)

# Evaluate SVM on test data
test_predictions <- predict(svm_model, as.matrix(test_data[, -ncol(test_data)]))
svm_conf_matrix <- confusionMatrix(test_predictions, as.factor(test_data[, ncol(test_data)]))

print("SVM Test Performance:")
print(svm_conf_matrix)

# Visualize the confusion matrix for test data
svm_conf_matrix_df <- as.data.frame(svm_conf_matrix$table)
ggplot(svm_conf_matrix_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  labs(title = "Confusion Matrix: Test Data (SVM)", x = "Actual", y = "Predicted") +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_minimal()