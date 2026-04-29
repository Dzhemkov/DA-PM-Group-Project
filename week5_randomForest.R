# install.packages("randomForest")
library(randomForest)
library(caret)

# ('train_set' and 'test_set' is needed from Week 4)
# Ensure NAs are gone
train_set <- na.omit(train_set)
test_set <- na.omit(test_set)

# Build the Random Forest Model
# We are predicting 'Survived' using the same features as before
set.seed(42) # For consistent results
rf_model <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare, 
                         data = train_set, 
                         ntree = 500,        # Number of trees in the forest
                         importance = TRUE)  # To see which variables matter most

# Model Summary
print(rf_model)

# Make Predictions on the Test Set
rf_preds <- predict(rf_model, test_set)

# Evaluate Performance
# Accuracy, Precision, and Recall for Week 5
rf_conf_matrix <- confusionMatrix(rf_preds, test_set$Survived, positive = "1")
print(rf_conf_matrix)

# Check Variable Importance
# This shows which features (like Sex or Age) were most important to the model
varImpPlot(rf_model, main = "Variable Importance for Titanic Survival")