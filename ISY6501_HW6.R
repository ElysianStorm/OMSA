# Load necessary libraries
library(ggplot2)
library(dplyr)
library(caret)
library(factoextra)

# Load the dataset
crime_data <- read.table("../datasets/ISY6501_HW6_USCrime.txt", header = TRUE)

# Check structure of the dataset
str(crime_data)

# Scale the data before applying PCA
crime_scaled <- scale(crime_data[,-ncol(crime_data)]) # Exclude Crime column from scaling

# Perform PCA
pca_result <- prcomp(crime_scaled, center = TRUE, scale. = TRUE)
summary(pca_result) # View variance explained by components

# Visualizing explained variance
fviz_eig(pca_result)

# Select first few principal components that explain most of the variance
num_pc <- which(cumsum(pca_result$sdev^2 / sum(pca_result$sdev^2)) > 0.95)[1]
pca_scores <- as.data.frame(pca_result$x[, 1:num_pc])
pca_scores$Crime <- crime_data$Crime

# Build regression model using selected principal components
pca_model <- lm(Crime ~ ., data = pca_scores)
summary(pca_model)

# Compute original variable coefficients from principal components
pc_rot <- pca_result$rotation[, 1:num_pc]  # Loadings of the selected PCs
beta_pc <- coef(pca_model)[-1]  # Regression coefficients from PCA model

# Convert principal component regression coefficients to original variable space
original_coefficients <- pc_rot %*% beta_pc

# Convert to a named vector for clarity
original_coefficients <- as.data.frame(original_coefficients)
rownames(original_coefficients) <- colnames(crime_scaled)
colnames(original_coefficients) <- "Coefficient"

# Display the coefficients in terms of original variables
print(original_coefficients)

# Compare with previous regression model
original_model <- lm(Crime ~ ., data = crime_data)
summary(original_model)

# Visualizing residuals
par(mfrow = c(1, 2))
plot(pca_model$residuals, main = "PCA Model Residuals", ylab = "Residuals", xlab = "Index", col = "blue")
plot(original_model$residuals, main = "Original Model Residuals", ylab = "Residuals", xlab = "Index", col = "red")

# Evaluate performance
pca_pred <- predict(pca_model, newdata = pca_scores)
original_pred <- predict(original_model, newdata = crime_data)

pca_rmse <- sqrt(mean((crime_data$Crime - pca_pred)^2))
original_rmse <- sqrt(mean((crime_data$Crime - original_pred)^2))

cat("PCA Model RMSE:", pca_rmse, "\n")
cat("Original Model RMSE:", original_rmse, "\n")
