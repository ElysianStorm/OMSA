# Load libraries
set.seed(123)
library(dplyr)
library(ggplot2)
library(lpSolve)

# -------- Step 1: Simulate Data --------
n <- 200
product_types <- paste0("Product_", 1:5)
data <- data.frame(
  product_type = sample(product_types, n, replace = TRUE),
  shelf_space = runif(n, 2, 20)
)

# Define true sales relationship with diminishing returns
coeffs <- c(15, 20, 18, 22, 17)
names(coeffs) <- product_types
data$sales <- mapply(function(pt, ss) {
  c <- coeffs[pt]
  noise <- rnorm(1, 0, 2)
  c * log(1 + ss) + noise
}, data$product_type, data$shelf_space)

# -------- Step 2: Model Sales vs Shelf Space --------
models <- lapply(split(data, data$product_type), function(df) {
  lm(sales ~ log(1 + shelf_space), data = df)
})

# Visualize the regression fits
ggplot(data, aes(x = shelf_space, y = sales, color = product_type)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", formula = y ~ log(1 + x), se = FALSE) +
  theme_minimal() +
  labs(title = "Sales vs. Shelf Space for Different Product Types")

# -------- Step 3: Linear Programming Optimization --------
# Constraints
min_space <- c(5, 5, 5, 5, 5)     # Minimum allowed per product type
max_space <- c(20, 20, 20, 20, 20) # Maximum allowed per product type
total_space <- 60                 # Total available shelf space

# Objective: Use regression slope coefficients as linear proxies
coefs <- sapply(models, function(m) coef(m)[2])  # Extract slope
objective <- coefs

# Define constraint matrix
A <- rbind(
  diag(5),        # Shelf space >= min
  diag(5),        # Shelf space <= max
  rep(1, 5)       # Total shelf space constraint
)
dir <- c(rep(">=", 5), rep("<=", 5), "=")
rhs <- c(min_space, max_space, total_space)

# Solve LP
lp_result <- lp("max", objective, A, dir, rhs)
optimal_space <- lp_result$solution
names(optimal_space) <- product_types

# -------- Step 4: Output and Visualization --------
# Print optimal allocation
print("Optimal Shelf Space Allocation:")
print(optimal_space)

# Plot optimal shelf space allocation
optimal_df <- data.frame(
  product_type = product_types,
  optimal_shelf_space = optimal_space
)

ggplot(optimal_df, aes(x = product_type, y = optimal_shelf_space, fill = product_type)) +
  geom_bar(stat = "identity", width = 0.6) +
  theme_minimal() +
  labs(title = "Optimal Shelf Space Allocation", y = "Shelf Space (sq. ft)")

# Optional: Check model significance
print(summary(models[[1]]))  # Print one model's summary

