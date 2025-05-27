# Load necessary libraries
library(ggplot2)
library(forecast)
library(tidyverse)
library(smooth)

# Step 1: Load and Explore Data
temps <- read.table("../datasets/ISY6501_HW3_Temps.txt", header = TRUE)
colnames_clean <- gsub("^X", "", colnames(temps))  
glimpse(temps)  # Check structure
summary(temps)  # Summary statistics

# Step 2: Convert Data into Time Series Format
temps_ts <- ts(temps[,-1], start = 1996, frequency = 122)  # Exclude 'DAY' column

# Step 3: Plot Raw Data
temps_long <- temps %>%
  pivot_longer(cols = -DAY, names_to = "Year", values_to = "Temperature") %>%
  mutate(Year = as.numeric(gsub("[^0-9]", "", Year)))

ggplot(temps_long, aes(x = DAY, y = Temperature, group = Year, color = factor(Year))) +
  geom_line(alpha = 0.6) +
  labs(title = "Daily High Temperatures (July–Oct, Atlanta)", x = "Day of Year", y = "Temperature (°F)") +
  theme_minimal()

# Create an empty vector to hold all data
all_data <- c()

# Loop through the years 1996 to 2015
for (year in 1996:2015) {
  # Extract the data for each year, assuming column names are like "X1996", "X1997", ...
  year_data <- temps[[paste0("X", year)]]
  
  # Append the data for the year to the all_data vector
  all_data <- c(all_data, year_data)
}

# Create the time series for the entire dataset (1996 to 2015)
temps_ts <- ts(all_data, start = c(1996, 1), end = c(2015, length(all_data) %% 365), frequency = 365)

# Print the created time series
print(temps_ts)

# Plot the time series
plot(temps_ts, main = "Temperature Time Series (1996 - 2015)", ylab = "Temperature", xlab = "Year")

# Apply Holt-Winters or other models
fit_hw <- HoltWinters(temps_ts, beta = TRUE, gamma = TRUE)
print(fit_hw)

# Forecasting
forecast_hw <- forecast(fit_hw, h = 30)
autoplot(forecast_hw) +
  ggtitle("Holt-Winters Exponential Smoothing on Temperature Data")

# Step 4b: Apply Smooth Package's es Function
fit_es <- es(temps_ts, model = "AAM", silent = FALSE)
print(fit_es)

forecast_es <- forecast(fit_es, h = 30)
autoplot(forecast_es) +
  ggtitle("Smooth Package Exponential Smoothing Forecast")

# Step 5: Identify Trend Shift Over Years
trend_data <- temps_long %>%
  group_by(Year) %>%
  summarize(avg_temp = mean(Temperature, na.rm = TRUE))

ggplot(trend_data, aes(x = Year, y = avg_temp)) +
  geom_line() +
  geom_point() +
  labs(title = "Trend of Average High Temperatures Over 20 Years", x = "Year", y = "Average Temperature (°F)") +
  theme_minimal()
