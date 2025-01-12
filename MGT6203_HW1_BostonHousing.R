# Import that data into R (using read.table() or read.csv() functions)
data <- read.csv("../datasets/MGT6203_HW1_BostonHousing.csv")

# Print the first 10 rows of the data set (using head() function)
print(head(data, 10))

# Show a summary of all variables using summary() function
summary(data)

# Calculate the mean value of medv variable
mean_medv <- mean(data[["medv"]])

# Plot a histogram of medv variable. Mark the mean value of medv on the plot by adding a vertical line
medv <- data[["medv"]]
hist(medv, xlab = "MEDV", breaks=50)
abline(v = mean_medv, col = 'red')

# Create a variable called cat.medv and add to the data frame. It is a categorical variable whose value equals 1 if the medv value for a tract is greater than $30,000 and equals 0 otherwise
data$cat.medv <- ifelse(data$medv > 30, 1, 0)
print(data)

# Calculate the mean of cat.medv
mean.cat_medv <- mean(data[["cat.medv"]])
# Meaning of mean.cat_medv
# It represents the skew/alignment of the distribution of housing prices. Where a value of 0.5 would mean its a normal distribution with equal portions of houses below $30000 and above $30000. Here, since the value is 0.166, it means that the normal distribution is skewed towards houses less than $30000 (i.e. there are more houses below $30000 than above $30000).

# Calculate the mean of cat.medv for the tracts that bound Charles River (i.e., chas==1), and save it to a variable
mean.cat_medv.ch_bound <- mean(data$cat.medv[data$chas == 1])

# Calculate the mean of cat.medv for the tracts that don’t bound Charles River (i.e., chas==0), and save it to a variable
mean.cat_medv.ch_unbound <- mean(data$cat.medv[data$chas == 0])

# Create a vector that comprises the two calculated mean values
vect_means <- c(mean.cat_medv.ch_bound, mean.cat_medv.ch_unbound)

# Plot a bar chart using the above created vector as the data. Label the two bars properly
barplot(vect_means, names.arg = c("CH Bound", "CH Unbound"), ylab = "Mean")

# Discuss what you can tell from the bar chart
# The average house value bound to Charles River is close to $30000 whereas the average house value not bound to Charles River is significantly lower at approx $15000. This analysis combined with the analysis of question 7 implies that river bound houses have higher prices since they are seen as exclusive or limited in nature.

# Create a side-by-side boxplot of medv over chas, that is, show the distributions of medv for the two groups with chas==1 and chas==0
boxplot(data$medv ~ data$chas)
# The boxplot reveals that in houses that are not bound to Charles River, there is a huge variation in the values. However, the concentration of the majority of houses (interquartile range) lie between the range of 15-25. In case of the houses bound to Charles River, the normal distribution is skewed towards the lower end (i.e. the interquartile range is towards the lower end, which ranges between 20-35.) This also implies that the distribution of values of houses bound to the river are more spread out compared to housers that are ot bound to the river.

# Create a scatter plot of medv (y-axis) versus lstat (x-axis)
plot(x=data$lstat, y=data$medv, xlab="LSAT", ylab="MEDV")

# Run a simple linear regression of medv on lstat. Show the regression results
lnr_reg_medv_lstat <- lm(medv ~ lstat, data=data)

# Add the regression line onto the scatter plot. Add a legend appropriately
abline(lnr_reg_medv_lstat, col='red')
legend("topright", legend=c("MEDV-LSTAT"), col=c("red"))
