# Load required libraries
library(kernlab)
library(kknn)

# Load the dataset
data <- read.delim("<Placeholder for dataset txt file path with header>", 
                   header = TRUE, sep = "\t", dec = ".")

# Handle missing values (drop rows with NA)
data <- na.omit(data)

# Scale continuous variables
continuous_columns <- c("A2", "A3", "A8", "A11", "A14", "A15")
data[continuous_columns] <- scale(data[continuous_columns])

# Set Up Predictors (Independent Values) and Response (Dependent Value) and size N
predictors <- as.matrix(data[, 1:10])
response <- as.factor(data[, 11])
n <- nrow(data)

# --------------------------------------------------------- #

# Train a SVM Model
# Initialize variables
C_values <- 10^seq(-3, 3, by = 0.2)  # From 0.001 to 1000
accuracies <- numeric(length(C_values))
best_weights <- NULL
best_intercept <- NULL
best_accuracy <- 0
best_C <- 0

# Loop through C values and compute accuracy, weights, and intercept
for (i in seq_along(C_values)) {
  C <- C_values[i]
  
  # Train SVM model
  model <- ksvm(predictors, response, type = "C-svc", kernel = "vanilladot", 
                C = C, scaled = TRUE)
  
  # Predict on training data
  predictions <- predict(model, predictors)
  
  # Calculate accuracy
  accuracies[i] <- sum(predictions == response) / n
  
  # Check if this model is the best so far
  if (accuracies[i] > best_accuracy) {
    best_accuracy <- accuracies[i]
    best_C <- C
    best_weights <- colSums(model@xmatrix[[1]] * model@coef[[1]])
    best_intercept <- -model@b
  }
}

# Print the best model details
cat("Best C:", best_C, "\n")
cat("Best Accuracy:", best_accuracy, "\n")
cat("Weights:", best_weights, "\n")
cat("Intercept:", best_intercept, "\n")

#Plot SVM Accuracy against C Hyper-parameter
plot(log10(C_values), accuracies, type = "b", pch = 19, col = "blue",
     xlab = "log10(C)", ylab = "Accuracy", 
     main = "SVM Accuracy vs Regularization Parameter (C)")

# --------------------------------------------------------- #

# Train a KNN model
# Initialize variables
k_values <- seq(3, 20, by = 1)
accuracies <- numeric(length(k_values))

# Loop over different k values
for (j in seq_along(k_values)) {
  k <- k_values[j]
  correct <- 0
  
  # Leave-one-out cross-validation (LOOCV)
  for (i in 1:n) {
    # Separate training and test sets
    train_data <- predictors[-i, ]
    train_response <- response[-i]
    test_data <- predictors[i, , drop = FALSE]  # Ensure test_data remains a matrix
    
    # KNN prediction
    model <- kknn(train_response ~ ., train=data.frame(train_data), 
                  test=data.frame(test_data), k = k, scale = TRUE)
    pred <- fitted(model)
    
    # Compare prediction with actual value
    if (pred == as.numeric(response[i]) - 1) {  # Match factor levels with binary labels
      correct <- correct + 1
    }
  }
  
  # Calculate accuracy
  accuracies[j] <- correct / n
}

# Find the best k
best_k <- k_values[which.max(accuracies)]
best_accuracy <- max(accuracies)

# Output results
list(k_values = k_values, accuracies = accuracies, best_k = best_k, 
     best_accuracy = best_accuracy)
plot(k_values, accuracies, type = "b", xlab = "k", ylab = "Accuracy", 
     main = "Accuracy vs k")