# Load required libraries
library(car) # for vif calculation

# Task 1: Import Data and Run a Linear Regression
# Import data
data <- read.csv("../datasets/MGT_6203_HW2_1_UsedCars.csv")

# Perform linear regression
model <- lm(Price ~ Age + KM + HP + Metallic + Automatic + CC + Doors + Gears + Weight, data = data)
sum_model <- summary(model)

# Show summary
print(sum_model)

# Calculate fitted values and residuals
fitted_values <- model$fitted.values
residuals <- model$residuals

# Create a data frame with original y, fitted y_hat, and residuals for first 10 observations
results <- data.frame(Original = data$Price[1:10],
                      Fitted = fitted_values[1:10],
                      Residuals = residuals[1:10])
print(results)

# Task 2: t-Statistic and p-Value
# Calculate t-statistics manually
t_stats <- coef(sum_model)[, "Estimate"] / coef(sum_model)[, "Std. Error"]
print(t_stats)

# Degrees of freedom
df <- as.numeric(sum_model$df[2]) # Correctly extract residual degrees of freedom

# Calculate critical value at 95% confidence level
critical_value <- qt(0.975, df)
print(critical_value)

# Calculate p-values manually
p_values <- 2 * pt(-abs(t_stats), df)
print(p_values)

# Compare manually calculated p-values and t-statistics with summary results
comparison <- data.frame(Summary_t_Stats = coef(sum_model)[, "t value"],
                         Manual_t_Stats = t_stats,
                         Summary_p_Values = coef(sum_model)[, "Pr(>|t|)"],
                         Manual_p_Values = p_values)
print(comparison)

# Identify significant variables
significant_vars <- rownames(coef(sum_model))[p_values < 0.05]
print(significant_vars)

# Task 3: R^2 and VIF
# Calculate R-squared manually
rss <- sum(residuals^2)
tss <- sum((data$Price - mean(data$Price))^2)
r_squared_manual <- 1 - rss / tss
print(r_squared_manual)

# Compare with R-squared from summary
r_squared_summary <- sum_model$r.squared
print(r_squared_summary)

# Calculate VIF
vif_values <- vif(model)
print(vif_values)

# Identify variable with largest VIF (likely Weight)
largest_vif_var <- names(which.max(vif_values))
print(largest_vif_var)

# Recalculate VIF manually for Weight
weight_model <- lm(Weight ~ Age + KM + HP + Metallic + Automatic + CC + Doors + Gears, data = data)
r_squared_weight <- summary(weight_model)$r.squared
vif_weight <- 1 / (1 - r_squared_weight)
print(vif_weight)

# Task 4: Model Comparison
# Remove insignificant variables
significant_formula <- as.formula(paste("Price ~", paste(significant_vars, collapse = " + ")))
new_model <- lm(significant_formula, data = data)
sum_new_model <- summary(new_model)
print(sum_new_model)

# Compare R-squared and Adjusted R-squared
r2_comparison <- data.frame(Model = c("Full", "Reduced"),
                            R_Squared = c(sum_model$r.squared, sum_new_model$r.squared),
                            Adjusted_R_Squared = c(sum_model$adj.r.squared, sum_new_model$adj.r.squared))
print(r2_comparison)

# Effects of certain variables from the better model
# Correctly assign the better model based on adjusted R-squared
if (sum_new_model$adj.r.squared > sum_model$adj.r.squared) {
  better_model <- new_model
} else {
  better_model <- model
}
better_model_summary <- summary(better_model)

# Effect of Age (1 year older)
age_effect <- coef(better_model)["Age"] * 12
print(age_effect)

# Effect of 10,000 KM
km_effect <- coef(better_model)["KM"] * 10000
print(km_effect)
