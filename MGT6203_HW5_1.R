# Load required libraries
library(mlogit)
library(dplyr)

# Load dataset
data <- read.csv("../datasets/MGT6203_HW5_1_CommuteMode.csv")

# Ensure 'choice' is numeric
data$choice <- as.numeric(data$choice)

# Convert dataset to mlogit format
data_mlogit <- mlogit.data(data, choice = "choice", shape = "long", 
                           alt.var = "mode", id.var = "id")

# Estimate Multinomial Logit Model
model <- mlogit(choice ~ cost + time, data = data_mlogit)

# Compute fitted values (predicted probabilities for all alternatives)
fitted_values <- fitted(model, outcome = FALSE)
cat("Fitted values (Predicted choice probabilities for all alternatives):\n")
print(head(fitted_values))

# Compute predicted choice probabilities
predicted_probs <- predict(model, type = "probabilities")
cat("\nPredicted choice probabilities:\n")
print(predicted_probs)

# Compute marginal effects manually
marginal_effects <- predicted_probs * (1 - predicted_probs)
cat("\nMarginal effects:\n")
print(marginal_effects)

# Compute mean cost and time for each mode using tapply
mean_cost_by_mode <- tapply(data$cost, data$mode, mean, na.rm = TRUE)
cat("\nMean cost by mode:\n")
print(mean_cost_by_mode)

# Compute mean values of explanatory variables
mean_values <- data %>%
  group_by(mode) %>%
  summarize(cost = mean(cost, na.rm = TRUE),
            time = mean(time, na.rm = TRUE))

# Predict probabilities at mean values
prob_mean <- predict(model, newdata = mean_values, type = "probabilities")
cat("\nPredicted choice probabilities at mean values:\n")
print(prob_mean)

# Compute marginal effects at mean values
marginal_effects_mean <- prob_mean * (1 - prob_mean)
cat("\nMarginal effects at mean values:\n")
print(marginal_effects_mean)

