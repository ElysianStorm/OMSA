# Load the data-set
data <- read.table("../datasets/ISY6501_HW2_Iris.txt", header = TRUE)

# Step 2: Prepare the data for clustering (exclude the species column)
iris_data <- iris[, -5]

# Step 3: Scale the data to standardize it
iris_data_scaled <- scale(iris_data)

# Step 4: Compute the total within-cluster sum of squares (WSS) for different k values
wss <- function(k, data) {
  kmeans_result <- kmeans(data, centers = k, nstart = 20)
  return(kmeans_result$tot.withinss)
}

# Step 4: Function to calculate misclassifications for a given k
calculate_misclassifications <- function(k, data, true_labels) {
  # Perform k-means clustering
  kmeans_result <- kmeans(data, centers = k, nstart = 20)
  
  # Create a contingency table
  confusion_table <- table(kmeans_result$cluster, true_labels)
  
  # Determine the total number of misclassified points
  max_in_clusters <- apply(confusion_table, 1, max)  # Maximum correctly classified points in each cluster
  total_points <- sum(confusion_table)              # Total number of points
  misclassifications <- total_points - sum(max_in_clusters)
  
  return(list(k = k, misclassifications = misclassifications, confusion_table = confusion_table))
}

# Step 5: Iterate over a range of k values (e.g., 2 to 10)
results <- lapply(2:10, function(k) calculate_misclassifications(k, iris_data_scaled, iris$Species))

# Step 6: Display the misclassifications and confusion tables for each k
cat("Misclassifications for different k:\n")
for (result in results) {
  cat("For k =", result$k, "Misclassifications:", result$misclassifications, "\n")
  print(result$confusion_table)
  cat("\n")
}

# Step 7: Extract misclassifications for plotting
k_values <- sapply(results, function(x) x$k)
misclassifications <- sapply(results, function(x) x$misclassifications)

# Step 8: Create a bar plot of misclassifications
barplot(
  misclassifications,
  names.arg = k_values,
  main = "Number of Misclassifications for Different k",
  xlab = "Number of Clusters (k)",
  ylab = "Number of Misclassifications",
  col = "skyblue",
  border = "blue"
)

# Step 9: Calculate WSS for a range of k values
k_values <- 1:10
wss_values <- sapply(k_values, function(k) wss(k, iris_data_scaled))

# Step 10: Create the Elbow Plot
plot(
  k_values, 
  wss_values, 
  type = "b", # Line and points
  pch = 19,   # Solid circle points
  col = "blue", 
  main = "Elbow Method for Optimal k",
  xlab = "Number of Clusters (k)",
  ylab = "Total Within-Cluster Sum of Squares (WSS)"
)

# Highlight the potential optimal k (optional, e.g., k=3 for Iris dataset)
abline(v = 3, col = "red", lty = 2)  # Dashed red vertical line
text(3, wss_values[3], "Optimal k", pos = 4, col = "red")

# Step 11: Perform k-means clustering with the optimal k (optional step, k = 3 for Iris dataset)
optimal_k <- 3
kmeans_result <- kmeans(iris_data_scaled, centers = optimal_k, nstart = 20)

# Step 12: Create and display the confusion matrix for optimal k
cat("Confusion matrix for k =", optimal_k, ":\n")
confusion_matrix <- table(kmeans_result$cluster, iris$Species)
print(confusion_matrix)

