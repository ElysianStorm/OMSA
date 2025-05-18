# Homework 8 – Part 1: Neural Networks

# Load required libraries
install.packages("neuralnet", dependencies = TRUE)
library(neuralnet)

# Task 1: Data Preparation

# 1. Load data
smarket <- read.csv("../datasets/MGT6203_HW8_1_Smarket.csv")

# 2. Remove Year (1st column), Today (8th), and Up (9th) from scaling
mydata <- mydata[-c(1, 8)]         # Step 1: remove Year and Today
up_column <- mydata$Up             # Step 2: separate Up
scaled_features <- scale(mydata[, -7])  # Step 3: scale predictors only
scaled_data <- as.data.frame(scaled_features)
scaled_data$Up <- up_column        # Step 4: add Up back

# Combine scaled data with Up column
scaled_data <- as.data.frame(scaled_features)
scaled_data$Up <- up_column

# 3. Split into training (80%) and testing (20%) sets
set.seed(1000)
sample_indices <- sample(1:nrow(scaled_data), 1000)
train_data <- scaled_data[sample_indices, ]
test_data <- scaled_data[-sample_indices, ]

# Task 2: Single-Layer Neural Network (using Lag1, Lag2 only)

# 4. Load pre-fitted single-layer neural network
load("../datasets/MGT6203_HW8_1_Smarket_nn1.Rda")  # This loads variable 'nn1'
ls()
nn1 <- nn
# 5. Plot the neural network
plot(nn1)

# 6. Retrieve and display weights
nn1$weights

# 7. Manually calculate node activations for first test record
# Input values
x1 <- test_data$Lag1[1]
x2 <- test_data$Lag2[1]

# Extract weights for easier access
w_hidden_1 <- nn1$weights[[1]][[1]][,1]  # weights to hidden node 1 (including bias)
w_hidden_2 <- nn1$weights[[1]][[1]][,2]  # weights to hidden node 2
w_output   <- nn1$weights[[1]][[2]]      # weights to output node

# Add bias input term (1)
input_vector <- c(1, x1, x2)

# Hidden layer activations
z1 <- sum(w_hidden_1 * input_vector)
a1 <- 1 / (1 + exp(-z1))  # sigmoid

z2 <- sum(w_hidden_2 * input_vector)
a2 <- 1 / (1 + exp(-z2))  # sigmoid

# Output layer (also logistic activation)
output_input <- sum(w_output * c(1, a1, a2))
output <- 1 / (1 + exp(-output_input))

cat("Manual Output (Prediction Probability):", output, "\n")

# 8. Verify using compute()
compute_out <- compute(nn1, test_data[1, c("Lag1", "Lag2")])
print(compute_out)

# Task 3: Multi-Layer Neural Network (all predictors)

# 9. Load pre-fitted multi-layer neural network
load("../datasets/MGT6203_HW8_1_Smarket_nn2.Rda")  # This loads variable 'nn2'
ls()
nn2 <- nn

# 10. Plot and get weights
plot(nn2)
nn2$weights

# 11. Predict probabilities using compute()
multi_probs <- compute(nn2, test_data[, c("Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume")])
print(multi_probs)
multi_preds <- multi_probs$net.result
print(multi_preds)
# 12. Convert probabilities to TRUE/FALSE using 0.5 cutoff
predicted_up_multi <- multi_preds > 0.5

# 13. Confusion matrix and accuracy
conf_matrix_multi <- table(Predicted = predicted_up_multi, Actual = test_data$Up)
print(conf_matrix_multi)

accuracy_multi <- sum(diag(conf_matrix_multi)) / sum(conf_matrix_multi)
cat("Neural Net Accuracy:", accuracy_multi, "\n")

# 14. Logistic Regression Benchmark
glm_model <- glm(Up ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume,
                 data = train_data, family = binomial)
summary(glm_model)
# Predict probabilities
glm_probs <- predict(glm_model, newdata = test_data, type = "response")

# Predict labels
predicted_up_glm <- glm_probs > 0.5

# Confusion matrix and accuracy
conf_matrix_glm <- table(Predicted = predicted_up_glm, Actual = test_data$Up)
print(conf_matrix_glm)

accuracy_glm <- sum(diag(conf_matrix_glm)) / sum(conf_matrix_glm)
cat("Logistic Regression Accuracy:", accuracy_glm, "\n")

# 15. Comparison
cat("Accuracy Comparison:\n")
cat("Neural Net:", round(accuracy_multi, 4), "\n")
cat("Logistic Regression:", round(accuracy_glm, 4), "\n")

