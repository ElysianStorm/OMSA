# Task 1: Import Data and set stringsAsFactors as TRUE
# Import data
data <- read.csv("../datasets/MGT_6203_HW2_2_UsedCars2.csv", stringsAsFactors = TRUE)
# View the structure of the data
str(data)

# Create a list of variables that are factors
factor_variables <- names(data)[sapply(data, is.factor)]

# Print the list of factor variables
print(factor_variables)

# Task 2: Run linear regression
model <- lm(Price ~ Age + KM + HP + Automatic + Gears + Weight + Color, data = data)

# Display the summary of the regression result
summary(model)

# Show how dummy variables are coded for the 'Color' variable
contrasts(data$Color)

# Extract the coefficients for the model
coefficients <- coef(model)

# Extract the p-values for the model
p_values <- summary(model)$coefficients[, 4]

# Extract the coefficients and p-values for the color variables (excluding the intercept)
color_coeffs <- coefficients[grep("^Color", names(coefficients))]
print(color_coeffs)

color_pvals <- p_values[grep("^Color", names(p_values))]
print(color_pvals)

# Task 3: Run the linear regression without Color and with the interaction term between Age and KM
model_interaction <- lm(Price ~ Age + KM + HP + Automatic + Gears + Weight + Age:KM, data = data)

# Show the summary of the regression model
summary(model_interaction)

# Extract p-val for AGE:KM interaction
p_val_age_km <- summary(model_interaction)$coefficients[,4]["Age:KM"]
print(p_val_age_km)

# Extract the coefficient for AGE:KM
coeff_age_km <- summary(model_interaction)$coefficients[,1]["Age:KM"]
print(coeff_age_km)

# Task 4: Scatterplot of KM vs Price
plot(data$KM, data$Price, 
     xlab = "KM", ylab = "Price", 
     main = "Scatterplot of KM vs Price", 
     pch = 19, col = "blue")

# Fit a polynomial regression (degree 4) on KM and Automatic
model_gam <- lm(Price ~ poly(KM, 4, raw = TRUE) + Automatic, data = data)

# Show the regression result summary
summary(model_gam)

# Generate a sequence of KM values for prediction
km.grid <- seq(from = min(data$KM), to = max(data$KM), by = 1000)

# Predict fitted values from the model
preds <- predict(model_gam, newdata = list(KM = km.grid, 
                                           Automatic = rep(mean(data$Automatic), length(km.grid))))

# Add the fitted curve to the plot
lines(km.grid, preds, col = "red", lwd = 2)