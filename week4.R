#WEEK 4
install.packages("caret")
library(caret)
titanic_data <- Titanic.Dataset

#Data Cleaning (Required for Baseline)
#Remove rows with missing values (NA) and set factors
titanic_clean <- na.omit(titanic_data) #Removes rows with missing Age
titanic_clean$Survived <- as.factor(titanic_clean$Survived)
titanic_clean$Sex <- as.factor(titanic_clean$Sex)
titanic_clean$Pclass <- as.factor(titanic_clean$Pclass)

#SPLIT DATASET (70% Train/30% Test)
set.seed(42) # Ensures results are the same every time you run it
index <- createDataPartition(titanic_clean$Survived, p = 0.70, list = FALSE)

train_set <- titanic_clean[index, ]
test_set  <- titanic_clean[-index, ]


#BUILD BASELINE MODELS
#Linear Regression (Numerical Outcome: Predicting 'Fare')
#Predict how much a passenger paid based on their details
lm_model <- lm(Fare ~ Pclass + Sex + Age + SibSp + Parch, data = train_set)

#Logistic Regression (Classification: Predicting 'Survived')
#Predict if a passenger lived (1) or died (0)
log_model <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch, 
                 data = train_set, family = "binomial")


#EVALUATE MODELS
#Evaluate Linear Regression (RMSE)
lm_predictions <- predict(lm_model, test_set)
lm_rmse <- RMSE(lm_predictions, test_set$Fare)
cat("Linear Regression RMSE:", lm_rmse, "\n")

#Evaluate Logistic Regression (Accuracy, Precision, Recall)
log_probs <- predict(log_model, test_set, type = "response")
log_preds <- ifelse(log_probs > 0.5, 1, 0)

#Create Confusion Matrix
conf_matrix <- confusionMatrix(as.factor(log_preds), test_set$Survived, positive = "1")
print(conf_matrix)

#Extract specific metrics for your report:
cat("Accuracy:", conf_matrix$overall['Accuracy'], "\n")
cat("Precision:", conf_matrix$byClass['Precision'], "\n")
cat("Recall:", conf_matrix$byClass['Recall'], "\n")


#INTERPRET MODEL COEFFICIENTS
#View the weights/importance of each variable
cat("\n--- Linear Regression Coefficients ---\n")
print(summary(lm_model)$coefficients)

cat("\n--- Logistic Regression Coefficients ---\n")
print(summary(log_model)$coefficients)