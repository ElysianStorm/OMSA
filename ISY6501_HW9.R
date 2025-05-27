library(FrF2)

set.seed(123)  # Ensures reproducibility
design <- FrF2(nruns = 16, nfactors = 10)
print(design)

library(ggplot2)
library(reshape2)

design_df <- as.data.frame(design)
design_df$House <- 1:16
design_melted <- melt(design_df, id.vars = "House")

ggplot(design_melted, aes(x = variable, y = factor(House), fill = factor(value))) +
  geom_tile() +
  scale_fill_manual(values = c("red", "blue"), labels = c("Not Included", "Included")) +
  theme_minimal() +
  labs(title = "Feature Combinations in Experimental Design",
       x = "Feature", y = "House")

# ---------------------

# R Code for Binomial Distribution
n <- 10  # Number of trials
p <- 0.5  # Probability of success
x <- 0:n
y <- dbinom(x, size=n, prob=p)
barplot(y, names.arg=x, col="blue", main="Binomial Distribution (n=10, p=0.5)",
        xlab="Number of Successes", ylab="Probability")


# ---------------------

# R Code for Geometric Distribution
p <- 0.2  # Probability of success
x <- 1:15  # Number of trials
y <- dgeom(x - 1, prob=p)
barplot(y, names.arg=x, col="green", main="Geometric Distribution (p=0.2)",
        xlab="Number of Trials Before Success", ylab="Probability")

# ---------------------

# R Code for Poisson Distribution
lambda <- 5  # Average rate of occurrence
x <- 0:15
y <- dpois(x, lambda=lambda)
barplot(y, names.arg=x, col="red", main="Poisson Distribution (λ=5)",
        xlab="Number of Events", ylab="Probability")

# ----------------------

# R Code for Exponential Distribution
lambda <- 1  # Rate parameter
x <- seq(0, 5, by=0.1)
y <- dexp(x, rate=lambda)
plot(x, y, type="l", col="purple", lwd=2, main="Exponential Distribution (λ=1)",
     xlab="Time Between Events", ylab="Density")

# ----------------------

# R Code for Weibull Distribution
shape <- 2  # Shape parameter
scale <- 1  # Scale parameter
x <- seq(0, 5, by=0.1)
y <- dweibull(x, shape=shape, scale=scale)
plot(x, y, type="l", col="orange", lwd=2, main="Weibull Distribution (shape=2, scale=1)",
     xlab="Time Until Failure", ylab="Density")
