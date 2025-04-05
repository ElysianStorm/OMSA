# Load necessary packages
library(MASS)

# Import data
forum_data <- read.csv("../datasets/MGT6203_ForumPosts.csv")

# Fit Poisson regression model
poisson_model <- glm(posts ~ totalPosts + readingRate + postStock + wknd, data = forum_data, family = poisson)
poisson_summary <- summary(poisson_model)
print(poisson_summary)

# Compute overdispersion statistic
overdispersion <- sum(residuals(poisson_model, type = "pearson")^2) / poisson_model$df.residual
print(overdispersion)

# Fit Negative Binomial regression model
nb_model <- glm.nb(posts ~ totalPosts + readingRate + postStock + wknd, data = forum_data)
nb_summary <- summary(nb_model)
print(nb_summary)

# Compare AIC and BIC
poisson_aic <- AIC(poisson_model)
poisson_bic <- BIC(poisson_model)
nb_aic <- AIC(nb_model)
nb_bic <- BIC(nb_model)

print(c("Poisson AIC" = poisson_aic, "Poisson BIC" = poisson_bic))
print(c("NB AIC" = nb_aic, "NB BIC" = nb_bic))

# Predict probabilities for k = 0:20
k <- 0:20
mean_values <- colMeans(forum_data[, c("totalPosts", "readingRate", "postStock", "wknd")])

lambda_hat <- exp(sum(coef(poisson_model) * c(1, mean_values)))
poisson_probs <- dpois(k, lambda_hat)

theta_hat <- nb_model$theta
mu_nb <- exp(sum(coef(nb_model) * c(1, mean_values)))
nb_probs <- dnbinom(k, size = theta_hat, mu = mu_nb)

# Plot probabilities
plot(k, poisson_probs, type = "b", col = "blue", pch = 19, xlab = "Number of Posts", ylab = "Probability", main = "Predicted Probabilities")
points(k, nb_probs, type = "b", col = "red", pch = 18)
legend("topright", legend = c("Poisson", "Negative Binomial"), col = c("blue", "red"), pch = c(19, 18))

