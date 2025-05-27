# ---------------------------
# Section 0: Setup & Library Loading
# ---------------------------
# Load required packages
library(tidyverse)      # For data manipulation and visualization
library(caret)          # For model training and evaluation
library(e1071)          # For SVM
library(class)          # For KNN
library(ggplot2)

# Set a seed for reproducibility
set.seed(123)

# ---------------------------
# Section 1: Data Loading and Cleaning
# ---------------------------
# The dataset is assumed to be in the working directory.
# Adjust the file path if needed.
data_raw <- read.csv("../datasets/ISY6501_HW10_Breast-Cancer_Wisconsin.data.txt", header = FALSE,
                     na.strings = "?", stringsAsFactors = FALSE)

# Assign column names according to the UCI Breast Cancer Wisconsin (Original) dataset:
# Attributes: ID, Clump Thickness, Uniformity of Cell Size, Uniformity of Cell Shape,
# Marginal Adhesion, Single Epithelial Cell Size, Bare Nuclei, Bland Chromatin,
# Normal Nucleoli, Mitoses, and Class.
colnames(data_raw) <- c("ID", "Clump_Thickness", "Cell_Size", "Cell_Shape",
                        "Marginal_Adhesion", "Single_Epithelial", "Bare_Nuclei",
                        "Bland_Chromatin", "Normal_Nucleoli", "Mitoses", "Class")

# Convert relevant columns to numeric (all except ID and Class might be numeric).
data <- data_raw %>% mutate_at(vars(-ID, -Class), as.numeric)

# Quick summary of the data and missing values count
summary(data)
sapply(data, function(x) sum(is.na(x)))

# Visualize the distribution of missing values in Bare_Nuclei (which is known to have NAs)
ggplot(data, aes(x = Bare_Nuclei)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  ggtitle("Distribution of Bare_Nuclei (Before Imputation)") +
  theme_minimal()

# ---------------------------
# Section 2: Imputation Methods
# ---------------------------

# For the purpose of these exercises, we assume that the only missing values occur in the Bare_Nuclei column.
# Verify:
missing_col <- sapply(data, function(x) sum(is.na(x)))
print(missing_col)

# ---- 2.1 Mean/Mode Imputation ----
# Since Bare_Nuclei is numeric, we can impute missing values with the mean.
data_mean_imp <- data
mean_bare <- mean(data_mean_imp$Bare_Nuclei, na.rm = TRUE)
data_mean_imp$Bare_Nuclei[is.na(data_mean_imp$Bare_Nuclei)] <- mean_bare

# Visualize imputed distribution
ggplot(data_mean_imp, aes(x = Bare_Nuclei)) +
  geom_histogram(binwidth = 1, fill = "lightgreen", color = "black") +
  ggtitle("Distribution of Bare_Nuclei (Mean Imputation)") +
  theme_minimal()

# ---- 2.2 Regression Imputation ----
# We predict Bare_Nuclei using the other predictors (excluding ID and Class).
# Use complete cases to train the regression model.
train_data <- data %>% filter(!is.na(Bare_Nuclei))
model_reg <- lm(Bare_Nuclei ~ Clump_Thickness + Cell_Size + Cell_Shape +
                  Marginal_Adhesion + Single_Epithelial + Bland_Chromatin +
                  Normal_Nucleoli + Mitoses, data = train_data)
summary(model_reg)

# Create a copy for regression imputation
data_reg_imp <- data

# Identify rows with missing Bare_Nuclei
missing_idx <- which(is.na(data_reg_imp$Bare_Nuclei))

# Predict missing values using the regression model
predicted_values <- predict(model_reg, newdata = data_reg_imp[missing_idx, ])
data_reg_imp$Bare_Nuclei[missing_idx] <- predicted_values

# Visualize the imputed values with regression
ggplot(data_reg_imp, aes(x = Bare_Nuclei)) +
  geom_histogram(binwidth = 1, fill = "lightblue", color = "black") +
  ggtitle("Distribution of Bare_Nuclei (Regression Imputation)") +
  theme_minimal()

# ---- 2.3 Regression with Perturbation Imputation ----
# Here, we add a random noise term to the regression predictions.
data_reg_pert_imp <- data
# Calculate residual standard error from the regression model
resid_se <- summary(model_reg)$sigma

# Predict missing values and add perturbation
predicted_values_pert <- predict(model_reg, newdata = data_reg_pert_imp[missing_idx, ]) +
  rnorm(length(missing_idx), mean = 0, sd = resid_se)
data_reg_pert_imp$Bare_Nuclei[missing_idx] <- predicted_values_pert

# Visualize the distribution after perturbation imputation
ggplot(data_reg_pert_imp, aes(x = Bare_Nuclei)) +
  geom_histogram(binwidth = 1, fill = "salmon", color = "black") +
  ggtitle("Distribution of Bare_Nuclei (Regression with Perturbation)") +
  theme_minimal()

# ---------------------------
# Section 3: Optional - Classification Model Comparison
# ---------------------------
# In this section, we build classification models to predict the "Class" variable using
# the different imputed datasets, as well as other strategies:
#
# (a) Mean imputation data: data_mean_imp
# (b) Regression imputation data: data_reg_imp
# (c) Regression with perturbation imputation data: data_reg_pert_imp
# (d) Data with complete cases only (removing rows with missing values)
# (e) Data with an additional binary indicator for missing Bare_Nuclei

# For demonstration, we will build two classifiers: SVM and KNN.
# We will use 10-fold cross-validation to compare their performance.

# Helper function for model training and evaluation
evaluate_model <- function(dataset, method_name) {
  # Prepare the data: remove ID column, and convert Class to factor
  dataset <- dataset %>% select(-ID)
  dataset$Class <- as.factor(dataset$Class)
  
  # Define trainControl for 10-fold CV
  ctrl <- trainControl(method = "cv", number = 10)
  
  # SVM Model
  svm_model <- train(Class ~ ., data = dataset, method = "svmRadial", trControl = ctrl)
  
  # KNN Model: using default k from caret tuning
  knn_model <- train(Class ~ ., data = dataset, method = "knn", trControl = ctrl)
  
  cat("-----", method_name, "-----\n")
  cat("SVM Accuracy:", svm_model$results$Accuracy[which.max(svm_model$results$Accuracy)], "\n")
  cat("KNN Accuracy:", knn_model$results$Accuracy[which.max(knn_model$results$Accuracy)], "\n\n")
}

# (a) Mean imputation
evaluate_model(data_mean_imp, "Mean Imputation")

# (b) Regression imputation
evaluate_model(data_reg_imp, "Regression Imputation")

# (c) Regression with perturbation
evaluate_model(data_reg_pert_imp, "Regression with Perturbation")

# (d) Complete cases (removing missing values)
data_complete <- data %>% filter(complete.cases(.))
evaluate_model(data_complete, "Complete Cases (Deletion)")

# (e) Missing Indicator: create a binary variable indicating whether Bare_Nuclei was missing originally
data_indicator <- data
data_indicator$BareMissing <- ifelse(is.na(data_indicator$Bare_Nuclei), 1, 0)
# Impute Bare_Nuclei with mean (or any method) for modeling, now keeping the indicator
data_indicator$Bare_Nuclei[is.na(data_indicator$Bare_Nuclei)] <- mean_bare
evaluate_model(data_indicator, "Mean Imputation + Missing Indicator")

# ---------------------------
# Section 3: Model Evaluation to compare and visualize
# ---------------------------
# Function to train and evaluate models
evaluate_model <- function(dataset, method_name) {
  dataset <- dataset %>% select(-ID)
  dataset$Class <- as.factor(dataset$Class)
  ctrl <- trainControl(method = "cv", number = 10)
  
  # Train SVM model
  svm_model <- train(Class ~ ., data = dataset, method = "svmRadial", trControl = ctrl)
  # Train KNN model
  knn_model <- train(Class ~ ., data = dataset, method = "knn", trControl = ctrl)
  
  return(data.frame(
    Method = method_name,
    SVM_Accuracy = svm_model$results$Accuracy[which.max(svm_model$results$Accuracy)],
    KNN_Accuracy = knn_model$results$Accuracy[which.max(knn_model$results$Accuracy)]
  ))
}

# Evaluate models for each dataset
results <- rbind(
  evaluate_model(data_mean_imp, "Mean Imputation"),
  evaluate_model(data_reg_imp, "Regression Imputation"),
  evaluate_model(data_reg_pert_imp, "Regression + Perturbation"),
  evaluate_model(data_complete, "Complete Cases")
)

# ---------------------------
# Section 4: Visualization of Model Accuracies
# ---------------------------

# Reshape data for visualization
results_long <- results %>%
  pivot_longer(cols = c(SVM_Accuracy, KNN_Accuracy), names_to = "Model", values_to = "Accuracy")

# Visualization: Comparing Accuracies
ggplot(results_long, aes(x = Method, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  geom_text(aes(label = round(Accuracy, 3)), position = position_dodge(width = 0.9), vjust = -0.5) +
  labs(title = "Model Accuracy Comparison across Imputation Methods",
       x = "Imputation Method", y = "Accuracy", fill = "Model") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

# ---------------------------
# End of Script
# ---------------------------
