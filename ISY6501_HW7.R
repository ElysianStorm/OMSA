# Load necessary libraries
library(tree)
library(randomForest)
library(caret)
library(ggplot2)
library(pROC)

# Load Crime Data
crime_data <- read.table("../datasets/ISY6502_HW7_USCrime.txt", header=TRUE)

# Split data into training and test sets
set.seed(42)
train_index <- createDataPartition(crime_data$Crime, p=0.8, list=FALSE)
train_data <- crime_data[train_index, ]
test_data <- crime_data[-train_index, ]

# Regression Tree Model
tree_model <- tree(Crime ~ ., data=train_data)
pred_tree <- predict(tree_model, test_data)
tree_rmse <- sqrt(mean((test_data$Crime - pred_tree)^2))

# Random Forest Model
rf_model <- randomForest(Crime ~ ., data=train_data, ntree=100)
pred_rf <- predict(rf_model, test_data)
rf_rmse <- sqrt(mean((test_data$Crime - pred_rf)^2))

# Feature Importance
importance <- importance(rf_model)
feature_imp <- data.frame(Feature = rownames(importance), Importance = importance[, 1])
ggplot(feature_imp, aes(x=reorder(Feature, Importance), y=Importance)) + 
  geom_bar(stat="identity", fill="steelblue") + coord_flip() +
  labs(title="Feature Importance in Random Forest", x="Feature", y="Importance")

# Load German Credit Data
german_credit <- read.table("../datasets/ISY6502_HW7_GermanCredit.txt", header=FALSE)
colnames(german_credit) <- c("Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings", "Employment", "InstallmentRate", "PersonalStatus", "OtherDebtors", "ResidenceYears", "Property", "Age", "OtherInstallment", "Housing", "ExistingCredits", "Job", "NumLiable", "Telephone", "ForeignWorker", "CreditRisk")

# Convert CreditRisk to binary (1 = Good, 0 = Bad)
german_credit$CreditRisk <- ifelse(german_credit$CreditRisk == 2, 0, 1)

# Convert categorical variables to factors
german_credit[sapply(german_credit, is.character)] <- lapply(german_credit[sapply(german_credit, is.character)], as.factor)

# Split data into training and test sets
set.seed(42)
train_index <- createDataPartition(german_credit$CreditRisk, p=0.8, list=FALSE)
train_data <- german_credit[train_index, ]
test_data <- german_credit[-train_index, ]

# Logistic Regression Model
logit_model <- glm(CreditRisk ~ ., data=train_data, family=binomial)
pred_probs <- predict(logit_model, test_data, type="response")
pred_classes <- ifelse(pred_probs > 0.5, 1, 0)
accuracy <- mean(pred_classes == test_data$CreditRisk)

# Optimize Threshold
thresholds <- seq(0.1, 0.9, length.out=50)
costs <- sapply(thresholds, function(t) {
  pred <- ifelse(pred_probs >= t, 1, 0)
  false_neg <- sum(test_data$CreditRisk == 1 & pred == 0)
  false_pos <- sum(test_data$CreditRisk == 0 & pred == 1)
  return(false_neg + 5 * false_pos)
})
optimal_threshold <- thresholds[which.min(costs)]

# Probability Distribution Plot
ggplot(data.frame(Prob=pred_probs, Actual=factor(test_data$CreditRisk)), aes(x=Prob, fill=Actual)) + 
  geom_histogram(position="identity", alpha=0.5, bins=20) +
  geom_vline(xintercept=optimal_threshold, linetype="dashed", color="black") +
  labs(title="Probability Distribution of Credit Risk Predictions", x="Predicted Probability", y="Frequency") +
  scale_fill_manual(values=c("red", "blue"), labels=c("Bad Credit", "Good Credit"))