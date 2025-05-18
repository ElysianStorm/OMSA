# --------------------------
# Power Shutoff Optimization
# Full R Script
# --------------------------

# Install required packages (if missing)
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  caret, xgboost, pROC, ggplot2, cluster, TSP, lpSolve, dplyr
)

# Set seed for reproducibility
set.seed(123)

# =====================================
# 1. Simulate Customer Data
# =====================================
customers <- data.frame(
  customer_id = 1:1000,
  days_overdue = sample(30:180, 1000, replace = TRUE),
  balance = runif(1000, 100, 5000),
  payment_prob = runif(1000, 0, 1),
  lat = runif(1000, 37.7, 37.8),
  lon = runif(1000, -122.5, -122.4)
)

# Simulate 'paid' status (target variable)
customers$paid <- ifelse(customers$payment_prob > 0.5, 1, 0)  # Binary outcome

# =====================================
# 2. Predictive Model (XGBoost)
# =====================================
# Split data into training/test sets
train_idx <- createDataPartition(customers$paid, p = 0.8, list = FALSE)
train <- customers[train_idx, ]
test <- customers[-train_idx, ]

# Train XGBoost model
xgb_model <- xgboost(
  data = as.matrix(train[, c("days_overdue", "balance", "payment_prob")]),
  label = train$paid,
  nrounds = 100,
  objective = "binary:logistic",
  verbose = 0
)

# Predict probabilities on test data
test$pred_prob <- predict(xgb_model, as.matrix(test[, c("days_overdue", "balance", "payment_prob")]))

# Generate ROC curve
roc_obj <- roc(test$paid, test$pred_prob)
png("roc_curve.png", width = 800, height = 600)
plot(roc_obj, main = "ROC Curve for Payment Prediction")
dev.off()

# Print AUC
cat("AUC:", auc(roc_obj), "\n")

# =====================================
# 3. Priority Scoring
# =====================================
customers$priority_score <- (1 - customers$payment_prob) * customers$balance

# Plot histogram of priority scores
png("priority_histogram.png", width = 800, height = 600)
ggplot(customers, aes(x = priority_score)) +
  geom_histogram(fill = "tomato", bins = 30) +
  labs(title = "Distribution of Shutoff Priority Scores")
dev.off()

# =====================================
# 4. Route Optimization (TSP)
# =====================================
# Cluster customers into 10 regions
clusters <- kmeans(customers[, c("lat", "lon")], centers = 10)
customers$cluster <- clusters$cluster

# Solve TSP for Cluster 1
cluster_data <- subset(customers, cluster == 1)
tsp_data <- ETSP(cluster_data[, c("lon", "lat")])
tour <- solve_TSP(tsp_data)

# Plot optimized route
png("tsp_route.png", width = 800, height = 600)
plot(tsp_data, tour, col = "blue", pch = 20, main = "Optimized TSP Route for Cluster 1")
dev.off()

# =====================================
# 5. Capacity Planning (Integer Programming)
# =====================================
# Example: Worker capacity = 480 minutes (8 hours)
job_time <- c(30, 45, 60, 25, 40)  # Time per job (minutes)
priority <- c(100, 200, 150, 300, 80)

# Solve optimization problem
result <- lp(
  direction = "max",
  objective.in = priority,
  const.mat = matrix(job_time, nrow = 1),
  const.dir = "<=",
  const.rhs = 480,
  all.bin = TRUE
)

# Print selected jobs
cat("Selected Jobs:", which(result$solution == 1), "\n")

