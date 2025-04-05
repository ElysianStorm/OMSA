# Load necessary libraries
library(ggplot2)
library(cluster)
library(factoextra)
library(dendextend)

# Load dataset
data <- read.csv("../datasets/MGT6203_HW6_1_ShoppingVisits.csv")

# Task 1: K-means Clustering and Scaling
set.seed(123)
kmeans_orig <- kmeans(data, centers = 3, nstart = 20)
print(kmeans_orig$cluster)

# Compute cluster means for 'money'
cluster_means_money <- aggregate(data$money, by = list(cluster = kmeans_orig$cluster), FUN = mean)
print(cluster_means_money)

# Scale the data
data_scaled <- scale(data)
kmeans_scaled <- kmeans(data_scaled, centers = 3, nstart = 20)
print(kmeans_scaled$cluster)

# Compute cluster means for 'visits'
cluster_means_visits <- aggregate(data_scaled[, "visits"], by = list(cluster = kmeans_scaled$cluster), FUN = mean)
print(cluster_means_visits)

# Scatter plots
par(mfrow = c(1, 2))
plot(data, col = kmeans_orig$cluster, main = "K-means Clustering (Original)")
plot(data, col = kmeans_scaled$cluster, main = "K-means Clustering (Scaled)")

# Task 2: K-means with Different Cluster Numbers
kmeans_2 <- kmeans(data_scaled, centers = 2, nstart = 20)
kmeans_4 <- kmeans(data_scaled, centers = 4, nstart = 20)
kmeans_5 <- kmeans(data_scaled, centers = 5, nstart = 20)

# Display scatter plots
par(mfrow = c(2, 2))
plot(data, col = kmeans_2$cluster, main = "2 Clusters")
plot(data, col = kmeans_scaled$cluster, main = "3 Clusters")
plot(data, col = kmeans_4$cluster, main = "4 Clusters")
plot(data, col = kmeans_5$cluster, main = "5 Clusters")

# Task 3: Elbow Method
wss <- sapply(1:5, function(k) kmeans(data_scaled, centers = k, nstart = 20)$tot.withinss)
print(data.frame(Clusters = 1:5, WSS = wss))
plot(1:5, wss, type = "b", pch = 19, main = "Elbow Chart", xlab = "Number of Clusters", ylab = "WSS")

# Task 4: Hierarchical Clustering
dist_matrix <- dist(data_scaled, method = "euclidean")

# Average linkage
hc_avg <- hclust(dist_matrix, method = "average")
plot(hc_avg, main = "Dendrogram (Average Linkage)")
abline(h = 5, col = "red", lty = 2)  # Adjust based on height for 3 clusters

# Task 5: Compare Hierarchical vs K-means
hc_clusters <- cutree(hc_avg, k = 3)
par(mfrow = c(1, 2))
plot(data, col = kmeans_scaled$cluster, main = "K-means (3 Clusters)")
plot(data, col = hc_clusters, main = "Hierarchical (3 Clusters)")

# Average linkage
hc_avg <- hclust(dist_matrix, method = "average")
plot(hc_avg, main = "Dendrogram (Average Linkage)")
abline(h = 5, col = "red", lty = 2)  # Adjust based on height for 3 clusters

# Extract and print merge heights
merge_heights <- hc_avg$height
print(merge_heights)

# Task 6: Different Linkages
hc_complete <- hclust(dist_matrix, method = "complete")
hc_single <- hclust(dist_matrix, method = "single")
hc_centroid <- hclust(dist_matrix, method = "centroid")

# Cut dendrograms to get cluster memberships
hc_complete_clusters <- cutree(hc_complete, k = 3)
hc_single_clusters <- cutree(hc_single, k = 3)
hc_centroid_clusters <- cutree(hc_centroid, k = 3)

# Scatter plots for different linkage methods
par(mfrow = c(2, 2))
plot(data[, 1:2], col = hc_clusters, pch = 19, main = "Average Linkage")
plot(data[, 1:2], col = hc_complete_clusters, pch = 19, main = "Complete Linkage")
plot(data[, 1:2], col = hc_single_clusters, pch = 19, main = "Single Linkage")
plot(data[, 1:2], col = hc_centroid_clusters, pch = 19, main = "Centroid Linkage")

# Task 7: Interpret Clusters
# Interpretation and marketing strategy discussion can be added separately

# Task 8: Variance of 'money' across different cluster solutions
variance_money <- function(kmeans_result) {
  cluster_variances <- aggregate(data$money, by = list(cluster = kmeans_result$cluster), FUN = var)
  mean(cluster_variances$x) # Return mean variance across clusters
}

var_2 <- variance_money(kmeans_2)
var_3 <- variance_money(kmeans_scaled)
var_4 <- variance_money(kmeans_4)
var_5 <- variance_money(kmeans_5)

print(c("Variance for 2 clusters" = var_2,
        "Variance for 3 clusters" = var_3,
        "Variance for 4 clusters" = var_4,
        "Variance for 5 clusters" = var_5))