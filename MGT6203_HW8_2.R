# -----------------------------------------
# Homework 8 – Part 2: Deep Learning (CNN)
# CIFAR-100 Classification with Pretrained Model
# -----------------------------------------

# ===== Task 1: Install required packages =====
# Uncomment if not already installed
install.packages("torch")
install.packages("luz")
install.packages("torchvision")

# ===== Task 2: Load and inspect the CIFAR-100 data =====
library(torch)
library(luz)
library(torchvision)

# Transform function to convert image to tensor
transform <- function(x) {
  transform_to_tensor(x)
}

# Load training and testing datasets
train_ds <- cifar100_dataset(root="./", train=TRUE, download=TRUE, transform=transform)
test_ds <- cifar100_dataset(root="./", train=FALSE, transform=transform)

# Load fine label names (100 categories)
labels <- read.table("../datasets/MGT6203_HW8_2_Fine_Label_Names.txt")
labels <- labels$V1

# Visualize first 25 images from training data
par(mar=c(0,0,0,0), mfrow = c(5, 5))
for (i in 1:25) 
  plot(as.raster(as.array(train_ds[i][[1]]$permute(c(2,3,1)))))

cat.indx <- sapply(1:25, function(x) train_ds[x][[2]])
print(matrix(labels[cat.indx], 5, 5, byrow=TRUE))

# ===== Task 3: Define CNN architecture and inspect structure =====

# Define one convolution-pooling block
conv_block <- nn_module(
  initialize = function(in_channels, out_channels) {
    self$conv <- nn_conv2d(
      in_channels = in_channels,
      out_channels = out_channels,
      kernel_size = c(3,3),
      padding = "same"
    )
    self$relu <- nn_relu()
    self$pool <- nn_max_pool2d(kernel_size = c(2,2))
  },
  forward = function(x) {
    x %>%
      self$conv() %>%
      self$relu() %>%
      self$pool()
  }
)

# Define CNN model with 4 conv-pool blocks and 2 dense layers
model <- nn_module(
  initialize = function() {
    self$conv <- nn_sequential(
      conv_block(3, 32),
      conv_block(32, 64),
      conv_block(64, 128),
      conv_block(128, 256)
    )
    self$output <- nn_sequential(
      nn_dropout(0.5),
      nn_linear(2*2*256, 512),
      nn_relu(),
      nn_linear(512, 100)
    )
  },
  forward = function(x) {
    x %>%
      self$conv() %>%
      torch_flatten(start_dim = 2) %>%
      self$output()
  }
)

# ===== Task 4: Load pre-trained model (skip training) =====
fitted <- luz_load("../datasets/MGT6203_HW8_2_cnn_cifar.Luz")

# Optional: visualize training progress (if available in the model object)
plot(fitted)

# ===== Task 5: Evaluate model on test set and inspect results =====

# Predict on test set
pred <- predict(fitted, test_ds)
pred.class <- torch_argmax(pred, dim=2)
pred.class <- as_array(pred.class)

# Get true labels
true.class <- sapply(1:10000, function(x) test_ds[x][[2]])

# Confusion matrix and accuracy
confusion <- table(pred.class, true.class)
accuracy <- sum(diag(confusion)) / sum(confusion)
print(paste("Out-of-sample accuracy:", round(accuracy, 4)))

# Get category names for image 1 and 24
true_1 <- labels[true.class[1] + 1]
pred_1 <- labels[pred.class[1] + 1]
true_24 <- labels[true.class[24] + 1]
pred_24 <- labels[pred.class[24] + 1]

cat("1st Image - True Label:", true_1, "| Predicted:", pred_1, "\n")
cat("24th Image - True Label:", true_24, "| Predicted:", pred_24, "\n")

# Optional: Visualize first 25 misclassified images
wrong.list <- which(pred.class != true.class)
par(mar=c(0,0,0,0), mfrow=c(5,5))
for (i in wrong.list[1:25]) 
  plot(as.raster(as.array(test_ds[i][[1]]$permute(c(2,3,1)))))

cat("True Labels of Misclassified Images:\n")
cat.indx <- true.class[wrong.list[1:25]]
print(matrix(labels[cat.indx], 5, 5, byrow=TRUE))

cat("Predicted Labels of Misclassified Images:\n")
cat.indx <- pred.class[wrong.list[1:25]]
print(matrix(labels[cat.indx], 5, 5, byrow=TRUE))
