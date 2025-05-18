# Step 1: Install and load required packages
install.packages("tm")
install.packages("SnowballC")
install.packages("lsa")

library(tm)
library(SnowballC)
library(lsa)

# Step 2: Import the data
mydata <- read.csv("../datasets/MGT6203_HW7_1_Ads.csv", stringsAsFactors = FALSE)

# Convert the first two columns into a corpus
corpus <- Corpus(DataframeSource(mydata[, 1:2]))

# Save the third column as the outcome variable
labels <- mydata$label

# Step 3: Text preprocessing
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)

# Step 4: Construct term-document matrix
tdm <- TermDocumentMatrix(corpus)
inspect(tdm[1:30, 1:5])  # Display first 30 terms (across first 5 docs for visualization)

# Step 5: Calculate TF-IDF values
tdm_tfidf <- weightTfIdf(tdm)

# Step 6: Latent Semantic Analysis (LSA) with 20 concepts
lsa_result <- lsa(tdm_tfidf, dim = 20)

# Convert the document-concept matrix (dk) to a data frame
doc_concepts <- as.data.frame(as.matrix(lsa_result$dk))

# Step 7: Prepare training and test sets
set.seed(1111)
train_indices <- sample(1:nrow(doc_concepts), 800)
test_indices <- setdiff(1:nrow(doc_concepts), train_indices)

train_data <- cbind(label = labels[train_indices], doc_concepts[train_indices, ])
test_data  <- cbind(label = labels[test_indices], doc_concepts[test_indices, ])

# Step 8: Run logistic regression
logit_model <- glm(label ~ ., data = train_data, family = binomial)
summary(logit_model)

# Step 9: Evaluate model performance on test set
pred_probs <- predict(logit_model, newdata = test_data, type = "response")
predicted_labels <- ifelse(pred_probs >= 0.5, 1, 0)

# Confusion matrix
conf_matrix <- table(Predicted = predicted_labels, Actual = test_data$label)
print(conf_matrix)

# Accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Prediction Accuracy:", round(accuracy, 4), "\n")
