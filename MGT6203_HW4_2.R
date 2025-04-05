library(survival)

# Load the data
data <- read.csv("../datasets/MGT6203_HW4_BankAttrition.csv")

# Define the survival object
surv_obj <- Surv(time = data$ChurnTime, event = 1 - data$Censored)

# Estimate Weibull proportional hazards model
weibull_model <- survreg(surv_obj ~ Age + Income + HomeVal + Tenure + DirectDeposit + Loan + NumAccounts + Dist + MktShare, data = data, dist = "weibull")

summary(weibull_model)
# Extract coefficients and scale parameter
delta_estimates <- coef(weibull_model)
scale_param <- weibull_model$scale

# Transform parameters to hazard model form
beta_estimates <- -delta_estimates / scale_param
shape_param <- 1 / scale_param

# Display results
print(beta_estimates)
print(shape_param)

# Plot hazard function
time_seq <- seq(0, max(data$ChurnTime, na.rm = TRUE), length.out = 100)
hazard_values <- (shape_param / scale_param) * (time_seq / scale_param)^(shape_param - 1)
plot(time_seq, hazard_values, type = "l", col = "blue", xlab = "Time", ylab = "Hazard Rate", main = "Hazard Function")

# Plot density function
density_values <- (shape_param / scale_param) * (time_seq / scale_param)^(shape_param - 1) * exp(-(time_seq / scale_param)^shape_param)
hist(data$ChurnTime[data$Censored == 0], probability = TRUE, col = "gray", border = "white", main = "Density Function with Histogram", xlab = "Time")
lines(time_seq, density_values, col = "red", lwd = 2)

