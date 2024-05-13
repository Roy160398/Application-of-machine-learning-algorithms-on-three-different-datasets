# Load required library
library(caret)

# Load the cleaned data from CSV
df <- read.csv("C:/Users/User/OneDrive/Documents/DMML/termDepositCleaned.csv")

# Extract predictors (features) and target variable
X <- df[, -which(names(df) == "deposit")] # Exclude the target column
y <- df$deposit

# Apply Min-Max scaling
preprocessParams <- preProcess(X, method = c("range"))
X_scaled <- predict(preprocessParams, newdata = X)

# Split the dataset into train and test sets
set.seed(42) # for reproducibility
trainIndex <- sample(1:nrow(df), 0.7 * nrow(df))
X_train <- X_scaled[trainIndex, ]
y_train <- y[trainIndex]
X_test <- X_scaled[-trainIndex, ]
y_test <- y[-trainIndex]

# Convert y_train to factor
y_train <- as.factor(y_train)
print(dim(X_train))
print(length(y_train))

# Perform oversampling on the training data
#library(ROSE)
#oversampled_data <- ovun.sample(y_train ~ ., data = data.frame(cbind(y_train, X_train)), method = "over", seed = 42)$data


# Extract the oversampled predictors and target variable
#X_train <- oversampled_data[, -1]
#y_train <- oversampled_data[, 1]








# Define hyperparameters
hyperparameters <- list(
  neighbors = 9
)

# Train the KNN model
knn_model <- train(
  x = X_train,
  y = y_train,
  method = "knn",
  trControl = trainControl(method = "none"),  # Disable cross-validation
  tuneGrid = data.frame(.k = hyperparameters$neighbors)
)

# Make predictions on the test set
predictions <- predict(knn_model, newdata = X_test)

# Evaluate the model
accuracy <- sum(predictions == y_test) / length(y_test)
print(paste("Testing Accuracy of KNN with best hyperparameters:", accuracy))

# Calculate confusion matrix
conf_matrix <- table(predictions, y_test)

# Calculate precision, recall, and F1-score
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)

print("Evaluation Metrics:")
print(paste("Testing Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1-Score:", f1_score))


library(randomForest)

# Set seed for reproducibility
set.seed(42)

# Train the Random Forest model with best hyperparameters
rf_model <- randomForest(
  x = X_train,
  y = y_train,
  ntree = 200,                # Number of trees
  mtry = sqrt(ncol(X_train)), # Number of features to consider at each split
  max_depth = 20,             # Maximum depth of the trees
  min_node_size = 2,          # Minimum number of samples required to be at a leaf node
  importance = TRUE,          # Compute variable importance
  replace = TRUE              # Sample with replacement
)

# Make predictions on the test set
predictions <- predict(rf_model, newdata = X_test)

# Evaluate the model
accuracy <- sum(predictions == y_test) / length(y_test)
print(paste("Testing Accuracy of Random Forest with best hyperparameters:", accuracy))

# Calculate other evaluation metrics
conf_matrix <- confusionMatrix(predictions, y_test)
class_report <- as.data.frame(conf_matrix$byClass)
roc_auc <- round(auc(roc(y_test, as.numeric(predictions))), 4)

# Print other evaluation metrics
cat("Confusion Matrix:\n")
print(conf_matrix$table)
cat("Classification Report:\n")
print(class_report)
cat("ROC AUC Score:", roc_auc)