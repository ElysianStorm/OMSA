library(qcc)
library(tidyr)

# Load the temperature data
temps_data <- read.table("../datasets/ISY6501_HW3_Temps.txt", header = TRUE, sep = "\t")

# Rename columns to include year names (1996 to 2015)
colnames(temps_data) <- c("DAY", as.character(1996:2015))

# Convert the data to a long format
temps_long <- gather(temps_data, key = "Year", value = "Temperature", -DAY)

# Ensure 'Temperature' is numeric and 'Year' is treated as a character
temps_long$Temperature <- as.numeric(temps_long$Temperature)
temps_long$Year <- as.character(temps_long$Year)

# Ensure 'DAY' is properly formatted as a Date object
temps_long$Date <- as.Date(paste(temps_long$DAY, temps_long$Year, sep = "-"), format = "%d-%b-%Y")

# Extract month and day from the Date column
temps_long$Month <- as.numeric(format(temps_long$Date, "%m"))
temps_long$Day <- as.numeric(format(temps_long$Date, "%d"))

# Filter data for July through October
summer_data <- subset(temps_long, Month %in% 7:10)

# Function to detect end of summer using CUSUM
detect_end_of_summer <- function(year_data) {
  # Apply CUSUM to detect downward shifts (cooling)
  cusum_result <- cusum(year_data$Temperature, decision.interval = 5, se.shift = 1)
  
  # Extract LOWER violations (indices where cooling starts)
  violations <- cusum_result$violations$lower
  
  # Find the first violation (if any)
  if (any(violations == 1)) {
    end_of_summer_index <- which(violations == 1)[1]
    end_of_summer_date <- year_data$DAY[end_of_summer_index]
  } else {
    end_of_summer_date <- NA  # No significant cooling detected
  }
  
  return(end_of_summer_date)
}

# Apply the function to each year and collect results
years <- unique(summer_data$Year)
end_of_summer_dates <- sapply(years, function(year) {
  year_data <- subset(summer_data, Year == year)
  detect_end_of_summer(year_data)
})

# Create a data frame with the end of summer dates
end_of_summer_df <- data.frame(
  Year = years,
  EndOfSummer = as.Date(end_of_summer_dates, format = "%d-%b")
)

# Remove rows with NA values
end_of_summer_df <- na.omit(end_of_summer_df)

# Calculate the mean temperature for each year
mean_temperatures <- tapply(summer_data$Temperature, summer_data$Year, mean)
mean_temp_df <- data.frame(
  Year = as.numeric(names(mean_temperatures)),
  MeanTemp = as.numeric(mean_temperatures)
)

# Ensure 'MeanTemp' is numeric and remove any NA values
mean_temp_df$MeanTemp <- as.numeric(mean_temp_df$MeanTemp)
mean_temp_df <- na.omit(mean_temp_df)

# Check the structure of the data before applying CUSUM
str(mean_temp_df$MeanTemp)

# Apply CUSUM to mean temperatures for the warming trend
cusum_result <- cusum(mean_temp_df$MeanTemp, decision.interval = 5, se.shift = 1)

# Plot the CUSUM result for the warming trend
plot(cusum_result, main = "CUSUM Chart for Warming Trend", ylab = "Cumulative Sum")

# Plot the mean temperatures over the years
plot(mean_temp_df$Year, mean_temp_df$MeanTemp, 
     type = "b", xlab = "Year", ylab = "Mean Temperature", 
     main = "Mean Summer Temperature Over Years", col = "red", pch = 19)