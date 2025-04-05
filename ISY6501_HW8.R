# Load necessary libraries
library(MASS)      # For stepwise regression
library(glmnet)    # For Lasso and Elastic Net
library(ggplot2)   # For visualizations
library(caret)     # For data preprocessing

# Load the dataset
crime_data <- read.table("../datasets/ISY6501_HW8_USCrime.txt", header = TRUE)

# Inspect the structure of the dataset
str(crime_data)
summary(crime_data)

# Define predictors and response variable
y <- crime_data$Crime
X <- as.matrix(crime_data[, -ncol(crime_data)]) # Convert to matrix for glmnet

# ---- Stepwise Regression ----
stepwise_model <- stepAIC(lm(Crime ~ ., data = crime_data), direction = "both")
summary(stepwise_model)

# ---- Lasso Regression ----
# Standardize data before applying Lasso
X_scaled <- scale(X)
lasso_model <- cv.glmnet(X_scaled, y, alpha = 1)
plot(lasso_model) # Visualization of cross-validation results

# Extract best lambda and model coefficients
best_lambda_lasso <- lasso_model$lambda.min
lasso_coefs <- coef(lasso_model, s = best_lambda_lasso)
print(lasso_coefs)

# ---- Elastic Net Regression ----
elastic_net_model <- cv.glmnet(X_scaled, y, alpha = 0.5) # alpha=0.5 balances Lasso and Ridge
plot(elastic_net_model)

# Extract best lambda and model coefficients
best_lambda_en <- elastic_net_model$lambda.min
en_coefs <- coef(elastic_net_model, s = best_lambda_en)
print(en_coefs)

# ---- Visualizing Variable Importance ----
coef_df <- data.frame(Feature = rownames(lasso_coefs), 
                      Lasso = as.vector(lasso_coefs), ElasticNet = as.vector(en_coefs))
coef_df <- coef_df[-1, ] # Remove intercept

ggplot(coef_df, aes(x = Feature)) +
  geom_bar(aes(y = abs(Lasso)), stat = "identity", fill = "blue", alpha = 0.7) +
  geom_bar(aes(y = abs(ElasticNet)), stat = "identity", fill = "red", alpha = 0.5) +
  coord_flip() +
  theme_minimal() +
  labs(title = "Feature Importance (Lasso vs Elastic Net)", y = "Coefficient Magnitude", x = "Features")

