# Load necessary libraries
library(ggplot2)
library(dplyr)
library(caret)

# Import the dataset
data <- read.csv("../datasets/MGT_6203_HW3_1_Loan.csv")

# Convert Education to a factor variable
data$Education <- as.factor(data$Education)

# Task 1: Linear Probability Model (LPM)
lpm_model <- lm(Loan ~ Income + Family + CCAvg + Education, data = data)
summary(lpm_model)

# Identify fitted values outside [0,1] range
fitted_values <- predict(lpm_model)
invalid_fitted <- data[which(fitted_values < 0 | fitted_values > 1), ]
print(head(invalid_fitted))

# Task 2: Logit Model
logit_model <- glm(Loan ~ Income + Family + CCAvg + Education, data = data, family = binomial)
summary(logit_model)

# Create confusion matrix & calculate Percent Correctly Predicted (PCP)
thresh <- mean(data$Loan)
predicted <- ifelse(predict(logit_model, type = "response") > thresh, 1, 0)
conf_matrix <- table(Predicted = predicted, Actual = data$Loan)
conf_matrix
pcp_overall <- sum(diag(conf_matrix)) / sum(conf_matrix)
pcp_yes <- conf_matrix[2,2] / sum(conf_matrix[,2])
pcp_no <- conf_matrix[1,1] / sum(conf_matrix[,1])

# Predict probability at given values (Mean values & Education=2)
x_means <- data.frame(Income = mean(data$Income), Family = mean(data$Family), 
                      CCAvg = mean(data$CCAvg), Education = factor(2, levels = levels(data$Education)))
pred_prob <- predict(logit_model, newdata = x_means, type = "response")

# Coefficients comparison
lpm_coeff <- coef(lpm_model)
logit_coeff <- coef(logit_model)
data.frame(LPM = lpm_coeff, Logit = logit_coeff)

# Compute partial effects for Logit Model at mean values
beta <- coef(logit_model)
xb <- beta[1] + beta[2] * mean(data$Income) + beta[3] * mean(data$Family) + beta[4] * mean(data$CCAvg) + beta[5] * 2
p_x <- exp(xb) / (1 + exp(xb))
partial_effects <- beta[-1] * p_x * (1 - p_x)

# Compare partial effects with LPM
lpm_partial_effects <- coef(lpm_model)[-1]
data.frame(LPM = lpm_partial_effects, Logit = partial_effects)

# Extracting the coefficient for Family from LPM
lpm_coeff_family <- coef(lpm_model)["Family"]
print(paste("LPM partial effect for Family:", lpm_coeff_family))

# Compute partial effect for Logit model
beta <- coef(logit_model)
# Predicted probability at mean values of X
xb <- beta[1] + beta[2] * mean(data$Income) + beta[3] * mean(data$Family) + beta[4] * mean(data$CCAvg) + beta[5] * 2
p_x <- exp(xb) / (1 + exp(xb))
# Partial effect for Family in Logit model
logit_partial_effect_family <- beta[3] * p_x * (1 - p_x)
print(paste("Logit partial effect for Family:", logit_partial_effect_family))

