## Code for Project:
## Deepak, Abdullah, Mariano

library(corrplot)      
library(ISLR2)
library(caret) 
library(olsrr)
library(leaps)
library(MASS)
library(e1071)
library(class)
library(leaps)
library(glmnet)
library(pls)
library(pROC)
library(ggplot2)
#graphics.off()
#Load Data
Data <- read.csv("diabetes.csv")

#Data Types
str(Data)
#Summary
summary(Data)

#Get correlation values
correlationvalue = cor(Data[,-9])
correlationvalue

#Check correlation value out of range(0.75). #Results none.
out_of_range <- findCorrelation(correlationvalue, .75)
out_of_range
  
#Plot Correlation Values
corrplot.mixed( correlationvalue, order = "hclust", tl.cex = .85, tl.pos="lt",diag="n")

#Corrleation pairs
pairs(Data)

#Fornmat columns from int to numeric to be consistent
Data$Pregnancies <- as.numeric(Data$Pregnancies)
Data$Glucose <- as.numeric(Data$Glucose)
Data$BloodPressure <- as.numeric(Data$BloodPressure)
Data$SkinThickness <- as.numeric(Data$SkinThickness)
Data$Insulin <- as.numeric(Data$Insulin )
Data$Age <- as.numeric(Data$Age)
Data$Outcome <- as.factor(Data$Outcome)

#Verify formatting performed
str(Data)

#Perform Scaling
Data$Pregnancies <- scale(Data$Pregnancies, center = TRUE, scale = TRUE)
Data$Glucose <- scale(Data$Glucose, center = TRUE, scale = TRUE)
Data$BloodPressure <- scale(Data$BloodPressure, center = TRUE, scale = TRUE)
Data$SkinThickness <- scale(Data$SkinThickness, center = TRUE, scale = TRUE)
Data$Insulin <- scale(Data$Insulin, center = TRUE, scale = TRUE)
Data$BMI <- scale(Data$BMI, center = TRUE, scale = TRUE)
Data$DiabetesPedigreeFunction <- scale(Data$DiabetesPedigreeFunction, center = TRUE, scale = TRUE)
Data$Age <- scale(Data$Age, center = TRUE, scale = TRUE)
summary(Data)
#Verify scaling performed
summary(Data)


#Check if Box Transformation can be applied. Results no variables could be transformed with boxcox.
attach(Data)
BoxCoxTrans(Pregnancies)
BoxCoxTrans(Glucose)
BoxCoxTrans(BloodPressure)
BoxCoxTrans(SkinThickness)
BoxCoxTrans(Insulin)
BoxCoxTrans(BMI)
BoxCoxTrans(DiabetesPedigreeFunction)
BoxCoxTrans(Age)

#Check NearZero Variables. Results none.
nearZero <- nearZeroVar((Data))
colnames(Data[nearZeroVar(Data)])

#Using Logistic Regression we can observe that SkinThickness, Insulin and Age are not significant
# based on a 5% significance level.
glm.Diabetes <- glm(Outcome ~ ., data=Data, family=binomial)
summary(glm.Diabetes)

#Histogram to check distribution of Outcome(0 or 1)
histogram(Data$Outcome, xlab = "Outcome",type = "count",main="Original")

Diabetes_yes <- subset(Data, Outcome == 1)
dim(Diabetes_yes)
#268 with Diabetes = 34.9%

Diabetes_no <- subset(Data, Outcome == 0)
dim(Diabetes_no)
#500 without Diabetes = 65.1%

set.seed(1234)
#Split Data 70/30 so that same distribution is maintained
Data_split <- createDataPartition(Data$Outcome, p=0.7, list=FALSE) 
#Train Data created
Data_Train <- Data[Data_split,] 
#Test Data created
Data_Test <- Data[-Data_split,]
#Response Test Data created
Data.Outcome <- as.factor(Data$Outcome[-Data_split])


#Logistic Regression -- only with significant predictors
glm.Diabetes_sig <- glm(Outcome ~ Pregnancies + Glucose + BloodPressure + BMI + DiabetesPedigreeFunction, data=Data, family=binomial,subset=Data_split)
summary(glm.Diabetes_sig)
glm.Diabetes_pred <- predict(glm.Diabetes_sig, Data_Test, type="response")

#AUC PLOT
roc_object <- roc(Data.Outcome, glm.Diabetes_pred)
roc_object
auc <- auc(roc_object)
auc
plot(roc_object, main="AUC = 0.85")

glm.pred <- rep("0", 230)
glm.pred[glm.Diabetes_pred > 0.5] <- c("1")
table(glm.pred, Data.Outcome)

# Logistic Regression 78.26% correct prediction, error=21.74%
mean(glm.pred == Data.Outcome)

##LDA -- only with significant predictors
lda.Datafit <- lda(Outcome ~ Pregnancies + Glucose + BloodPressure + BMI + DiabetesPedigreeFunction, data=Data, family=binomial, subset=Data_split)
lda.Datafit

lda.pred <- predict(lda.Datafit, Data_Test, type="response")
lda.Dataclass <- lda.pred$class
table(lda.Dataclass,Data.Outcome)
 
mean(lda.Dataclass == Data.Outcome)


##Naives
nb.Datafit <- naiveBayes(Outcome ~ Pregnancies + Glucose + BloodPressure + BMI + DiabetesPedigreeFunction, data=Data, family=binomial,subset=Data_split)
nb.Datafit

nb.Dataclass <- predict(nb.Datafit, Data_Test)
table(nb.Dataclass, Data.Outcome)

mean(nb.Dataclass == Data.Outcome)



## KNN K=1
Datatrain.X <- as.matrix(cbind(Data$Pregnancies,Data$Glucose,Data$BloodPressure,Data$BMI,Data$DiabetesPedigreeFunction)[Data_split,])
Datatest.X <-  as.matrix(cbind(Data$Pregnancies,Data$Glucose,Data$BloodPressure,Data$BMI,Data$DiabetesPedigreeFunction)[-Data_split,])
Data.train.Outcome <- Data$Outcome[Data_split]

set.seed(125)
knn.Datapred <- knn(Datatrain.X, Datatest.X, Data.train.Outcome,k=1)
table(knn.Datapred, Data.Outcome)

mean(knn.Datapred == Data.Outcome)

## KNN K=10
set.seed(124)
knn.Datapred_10 <- knn(Datatrain.X, Datatest.X, Data.train.Outcome,k=10)
table(knn.Datapred_10, Data.Outcome)

mean(knn.Datapred_10 == Data.Outcome)

## KNN K=50
set.seed(123)
knn.Datapred_50 <- knn(Datatrain.X, Datatest.X, Data.train.Outcome,k=50)
table(knn.Datapred_50, Data.Outcome)

mean(knn.Datapred_50 == Data.Outcome)


## QDA
qda.fit <- qda(Outcome ~ Pregnancies + Glucose + BloodPressure + BMI + DiabetesPedigreeFunction, data=Data, family=binomial,subset=Data_split)
qda.fit

qda.class <- predict(qda.fit, Data_Test)$class
table(qda.class, Data.Outcome)
mean(qda.class == Data.Outcome)



##LASSO 
set.seed(12345)
x.lasso_Diabetes = model.matrix(Outcome ~ Pregnancies + Glucose + BloodPressure + BMI + DiabetesPedigreeFunction, Data_Test)[,-9]
#Let's get out Y
y.lasso_Diabetes= as.factor(Data_Test[,"Outcome"])

lasso_Diabetes <- cv.glmnet(x.lasso_Diabetes, y.lasso_Diabetes, alpha=1, family="binomial",type="class")
#Get lambda
minimum_lambda_Diabetes_lasso <- lasso_Diabetes$lambda.min
minimum_lambda_Diabetes_lasso
#Get lambda within 1 standard deviation
one_std_lambda_Diabetes_lasso <- lasso_Diabetes$lambda.1se
one_std_lambda_Diabetes_lasso

plot(lasso_Diabetes)

err_lasso = lasso_Diabetes$cvm[lasso_Diabetes$lambda == lasso_Diabetes$lambda.1se]
err_lasso


##RIDGE
set.seed(1234)
x.ridge_Diabetes = model.matrix(Outcome ~ Pregnancies + Glucose + BloodPressure + BMI + DiabetesPedigreeFunction, Data_Test)[,-9]
#Let's get out Y
y.ridge_Diabetes= as.factor(Data_Test[,"Outcome"])

ridge_Diabetes <- cv.glmnet(x.ridge_Diabetes, y.ridge_Diabetes, alpha=0, family="binomial",type.measure = "class")
#Get lambda
minimum_lambda_Diabetes_ridge <- ridge_Diabetes$lambda.min
minimum_lambda_Diabetes_ridge
#Get lambda within 1 standard deviation
one_std_lambda_Diabetes_ridge <- ridge_Diabetes$lambda.1se
one_std_lambda_Diabetes_ridge
plot(ridge_Diabetes)
err_ridge = ridge_Diabetes$cvm[ridge_Diabetes$lambda == ridge_Diabetes$lambda.1se]
err_ridge

#Elastic Net
x.enet_Diabetes = model.matrix(Outcome ~ Pregnancies + Glucose + BloodPressure + BMI + DiabetesPedigreeFunction, Data_Test)[,-9]
#Let's get out Y
y.enet_Diabetes= as.factor(Data_Test[,"Outcome"])
cv.enet = cv.glmnet(x.enet_Diabetes,y.enet_Diabetes,alpha = 0.5,
                       family = "binomial", type.measure = "class")

#Get lambda within 1 standard deviation
set.seed(123456)
one_std_lambda_Diabetes_enet <- cv.enet$lambda.1se
one_std_lambda_Diabetes_enet
plot(cv.enet)
err_enet = cv.enet$cvm[cv.enet$lambda == cv.enet$lambda.1se]
err_enet

 

 




