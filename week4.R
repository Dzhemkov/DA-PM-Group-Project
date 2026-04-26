#WEEK 4
library(caret)

# We use data_set which is already loaded in your environment
titanic_data <- data_set

#Data Cleaning (Required for Baseline)
#Remove rows with missing values (NA) and set factors using lowercase column names
titanic_clean <- na.omit(titanic_data) #Removes rows with missing values
titanic_clean$survived <- as.factor(titanic_clean$survived)
titanic_clean$sex <- as.factor(titanic_clean$sex)
titanic_clean$pclass <- as.factor(titanic_clean$pclass)

#SPLIT DATASET (70% Train/30% Test)
set.seed(42) # Ensures results are the same every time you run it
index <- createDataPartition(titanic_clean$survived, p = 0.70, list = FALSE)

# Force the matrix into a simple vector so R can process the negative sign
index_vec <- as.vector(index)

train_set <- titanic_clean[index_vec, ]
test_set  <- titanic_clean[-index_vec, ]


#BUILD BASELINE MODELS
#Linear Regression (Numerical Outcome: Predicting 'fare')
lm_model <- lm(fare ~ pclass + sex + age + sib_sp + parch, data = train_set)

#Logistic Regression (Classification: Predicting 'survived')
log_model <- glm(survived ~ pclass + sex + age + sib_sp + parch, 
                 data = train_set, family = "binomial")


#EVALUATE MODELS
#Evaluate Linear Regression (RMSE)
lm_predictions <- predict(lm_model, test_set)
lm_rmse <- RMSE(lm_predictions, test_set$fare)
cat("Linear Regression RMSE:", lm_rmse, "\n")

#Evaluate Logistic Regression (Accuracy, Precision, Recall)
log_probs <- predict(log_model, test_set, type = "response")
log_preds <- ifelse(log_probs > 0.5, 1, 0)

#Create Confusion Matrix
conf_matrix <- confusionMatrix(as.factor(log_preds), test_set$survived, positive = "1")
print(conf_matrix)

#Extract specific metrics for your report:
cat("\nAccuracy:", conf_matrix$overall['Accuracy'], "\n")
cat("Precision:", conf_matrix$byClass['Precision'], "\n")
cat("Recall:", conf_matrix$byClass['Recall'], "\n")


#INTERPRET MODEL COEFFICIENTS
cat("\n--- Linear Regression Coefficients ---\n")
print(summary(lm_model)$coefficients)

cat("\n--- Logistic Regression Coefficients ---\n")
print(summary(log_model)$coefficients)