# Load necessary libraries
install.packages("recommenderlab", dependencies = TRUE)
library(recommenderlab)

##### Task 1: Manual Calculation #####

# Given user-item matrix
ratings <- matrix(c(2, NA, 3, 5, 2, NA, 4, 4, 1), nrow=3, byrow=TRUE)
rownames(ratings) <- c("U1", "U2", "U3")
colnames(ratings) <- c("I1", "I2", "I3")

# Step i: Normalize by subtracting user mean
user_means <- rowMeans(ratings, na.rm = TRUE)
norm_ratings <- sweep(ratings, 1, user_means, FUN="-")

# Step ii: Compute cosine similarity between users
cos_sim <- function(x, y) {
  sum(x * y, na.rm = TRUE) / (sqrt(sum(x^2, na.rm = TRUE)) * sqrt(sum(y^2, na.rm = TRUE)))
}

s12 <- cos_sim(norm_ratings["U1", ], norm_ratings["U2", ])
s13 <- cos_sim(norm_ratings["U1", ], norm_ratings["U3", ])

# Step iii: Predict rating for U1 on I2
numerator <- s12 * norm_ratings["U2", "I2"] + s13 * norm_ratings["U3", "I2"]
denominator <- abs(s12) + abs(s13)
predicted_rating <- user_means["U1"] + (numerator / denominator)

# Explanation of the calculations
cat("Manual Calculation Process:\n")
cat("Step i: Normalize by subtracting user mean\n")
print(norm_ratings)

cat("Step ii: Compute cosine similarity:\n")
cat("s12 (similarity between U1 and U2):", s12, "\n")
cat("s13 (similarity between U1 and U3):", s13, "\n")

cat("Step iii: Predict rating for U1 on I2 using:\n")
cat("Formula: predicted_rating = user_mean(U1) + (s12 * rating(U2, I2) + s13 * rating(U3, I2)) / (|s12| + |s13|)\n")
cat("Predicted rating for U1 on I2:", predicted_rating, "\n")

##### Task 2: MovieLense Data #####

# Load the dataset
data(MovieLense)

fix(MovieLenseMeta)
fix(MovieLenseUser)
getRatingMatrix(MovieLense)

myrating <- matrix(NA, 1, 1664)
myrating[c(1,2,3,4,5)] <- c(3,5,5,4,4)
myrating <- as(myrating, "realRatingMatrix")

rec.ub <- Recommender(MovieLense, "UBCF")
pred.ub <- predict(rec.ub, myrating, n=10, type="topNList")
as(pred.ub, "list")
