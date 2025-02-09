# Load necessary libraries
library(ggplot2)
library(censReg)

# Load the dataset
data <- read.csv("../datasets/MGT6203_HW3_2_Mobile_data_usage.csv")

# Task 1: Scatter plot of DataUse vs Quota
plot1 <- ggplot(data, aes(x = Quota, y = DataUse)) +
  geom_point(alpha = 0.5) +
  labs(title = "Scatter plot of DataUse vs Quota", x = "Remaining Quota (MB)", y = "Daily Data Usage (MB)") +
  theme_minimal()
print(plot1)

# Explanation: The scatter plot may show a non-linear pattern, where data usage is lower when quota is low, suggesting censoring.

# Task 2: Linear regression model
lm_model <- lm(DataUse ~ Quota + Days, data = data)
summary(lm_model)

# Explanation: This model assumes a linear relationship between DataUse and Quota while controlling for Days.

# Task 3: Tobit model
cens_model <- censReg(DataUse ~ Quota + Days, data = data, left = 0)
summary(cens_model)

# Explanation: The Tobit model accounts for censored data where DataUse cannot be negative.

# Task 4: Compare coefficients
coefficients_lm <- coef(lm_model)[1:3]
coefficients_tobit <- coef(cens_model)[1:3]
comparison <- data.frame(Model = c("Linear", "Tobit"), 
                         Intercept = c(coefficients_lm[1], coefficients_tobit[1]),
                         Quota = c(coefficients_lm[2], coefficients_tobit[2]),
                         Days = c(coefficients_lm[3], coefficients_tobit[3]))
print(comparison)

# Explanation: The coefficient of Quota in the Tobit model differs from the linear model due to censoring effects.

# Task 5: Compute marginal effects
mean_days <- mean(data$Days, na.rm = TRUE)
marg_eff_10 <- margEff(cens_model, xValues = c(1, 10, mean_days))
marg_eff_2000 <- margEff(cens_model, xValues = c(1, 2000, mean_days))
print(marg_eff_10)
print(marg_eff_2000)

# Explanation: The marginal effect of Quota varies depending on its value due to the censoring nature of the model.
# Lower quotas may limit data usage, while higher quotas allow more consumption.