# 1. Install and load required packages
packages <- c("tm", "SnowballC", "topicmodels", "wordcloud")
installed <- packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])
lapply(packages, library, character.only = TRUE)

# 2. Import the data and convert to corpus
textdata <- read.csv("../datasets/MGT6203_HW7_2_News.csv", stringsAsFactors = FALSE)
corpus_original <- Corpus(DataframeSource(textdata))

# Step 3: Preprocessing
corpus_processed <- tm_map(corpus_original, stripWhitespace)
corpus_processed <- tm_map(corpus_processed, removePunctuation)
corpus_processed <- tm_map(corpus_processed, removeNumbers)
corpus_processed <- tm_map(corpus_processed, removeWords, stopwords("english"))
corpus_processed <- tm_map(corpus_processed, stemDocument)

# Step 4: Create Document-Term Matrix with min term frequency of 3
dtm <- DocumentTermMatrix(corpus_processed, control = list(bounds = list(global = c(3, Inf))))
dim(dtm)

# Step 5: Perform LDA with k = 20 topics, method = Gibbs
set.seed(1000)
lda_model <- LDA(dtm, k = 20, method = "Gibbs",
                 control = list(iter = 1000, verbose = 50, seed = 1000))

# Step 6: Extract posterior distributions
posterior_result <- posterior(lda_model)

# 6a: Retrieve terms matrix and its dimensions
terms_matrix <- posterior_result$terms
dim(terms_matrix)
head(terms_matrix[, 1:5])

# 6b: Retrieve topics matrix and its dimensions
topics_matrix <- posterior_result$topics
dim(topics_matrix)
head(topics_matrix[1:5, ])

# Step 7: Display top 10 terms per topic
top_terms <- terms(lda_model, 10)
print(top_terms)

# Step 8: Analyze one specific document (e.g., ID = 1082)
doc_id <- 1082
content(corpus_original[[doc_id]])  # Display full text of original article

# Barplot of topic distribution for selected document
topic_dist <- topics_matrix[doc_id, ]
barplot(topic_dist, main = paste("Topic Distribution for Document", doc_id),
        xlab = "Topic", ylab = "Probability")

# Word cloud for most relevant topic
top_topic <- which.max(topic_dist)
top_topic_terms <- sort(terms_matrix[top_topic, ], decreasing = TRUE)[1:50]
wordcloud(names(top_topic_terms), freq = top_topic_terms,
          scale = c(4, 0.5), max.words = 50, colors = brewer.pal(8, "Dark2"))
