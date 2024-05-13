
# Load required library
library(caret)

# Load the cleaned data from CSV
df <- read.csv("C:/Users/User/OneDrive/Documents/DMML/AdultCensusCleaned.csv")

# Extract predictors (features) and target variable
X <- df[, -which(names(df) == "income")] # Exclude the target column
y <- df$income

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

# Perform random oversampling on the training data
oversampled_data <- ovun.sample(y_train ~ ., data = data.frame(cbind(y_train, X_train)), method = "over", seed = 42)$data
X_train <- oversampled_data[, -1]
y_train <- oversampled_data[, 1]

# Define hyperparameters
hyperparameters <- list(
  neighbors = 1
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




# Convert y_train to factor
y_train <- as.factor(y_train)





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
  min_node_size = 1,          # Minimum number of samples required to be at a leaf node
  importance = TRUE,          # Compute variable importance
  replace = TRUE              # Sample with replacement
)

# Make predictions on the test set
predictions <- predict(rf_model, newdata = X_test)

# Evaluate the model
accuracy <- sum(predictions == y_test) / length(y_test)
print(paste("Testing Accuracy of Random Forest with best hyperparameters:", accuracy))