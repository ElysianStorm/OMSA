# Load necessary libraries
library(ggplot2)
library(dplyr)
library(caret)
library(car)

# Load the dataset
crime_data <- read.table("../datasets/ISY6501_HW5_USCrime.txt", header=TRUE)

# Explore the dataset
str(crime_data)
summary(crime_data)

# Question 8.1: Real-world example of linear regression
# Example: Predicting house prices based on factors such as square footage, number of bedrooms, 
# number of bathrooms, location, and year built. These predictors influence the house price, 
# making linear regression a suitable model.

# Question 8.2: Regression model for crime rate prediction

# Define the formula for regression
formula <- Crime ~ M + So + Ed + Po1 + Po2 + LF + M.F + Pop + NW + U1 + U2 + Wealth + Ineq + Prob + Time

# Fit the linear regression model
lm_model <- lm(formula, data=crime_data)

# Display the summary of the model
summary(lm_model)

# Check for multicollinearity
vif_values <- vif(lm_model)
print(vif_values)

# Predict crime rate for given city data
new_city <- data.frame(M = 14.0, So = 0, Ed = 10.0, Po1 = 12.0, Po2 = 15.5, 
                       LF = 0.640, M.F = 94.0, Pop = 150, NW = 1.1, 
                       U1 = 0.120, U2 = 3.6, Wealth = 3200, Ineq = 20.1, 
                       Prob = 0.04, Time = 39.0)

predicted_crime <- predict(lm_model, newdata = new_city)
print(predicted_crime)

# Visualization of model predictions
crime_data$predicted <- predict(lm_model)
ggplot(crime_data, aes(x=predicted, y=Crime)) + 
  geom_point() + 
  geom_smooth(method='lm', col='red') + 
  labs(title='Actual vs. Predicted Crime Rates', x='Predicted Crime Rate', y='Actual Crime Rate')

# Model evaluation
par(mfrow=c(2,2))
plot(lm_model)