# Load necessary library
library(outliers)

# Load the data
crime_data <- read.table("../datasets/ISY6501_HW3_USCrime.txt", header = TRUE)

# Extract the last column
crime_rates <- crime_data[, ncol(crime_data)]

# View structure and last column summary
summary(crime_data$Crime)

# Perform Grubbs' test
grubbs_test_result <- grubbs.test(crime_rates)

# Print the result
print(grubbs_test_result)

# Plot the data to visualize outliers
boxplot(crime_rates, main = "Boxplot of Crime Rates", ylab = "Crimes per 100,000 people")