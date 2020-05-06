## Author: Hallie Edwards
## Project Title: Spotify Audio Features Machine Learning Exploration
## Last Updated: 5-6-2020


# Initial Notes and Links
## Notes
#### This was done for a project for my BSAN 400 Intro to Machine Learning with R class. The goal of this project is to use audio features from Spotify to predict whether I will like a song or not. The data was pulled from two playlists I made and contains 860 songs total (460 like, 400 dislike). I was intentional about adding songs from various genres into both and included songs from the same artist in both playlists.

#### The Data was pulled using Spotify's Web API. I used Spotipy in python to pull the data and assign 1 or 0 to the "Like" column. Then I exported it and imported it here.

## Layout of file
#### Starts with basic EDA, then moves to basic generalized linear models (using the full model, a reduced model, and then models based on forward/backward/stepwise regression with respect to AIC and then respect to BIC), then it moves on to advanced logit models (Bagging, Classification Trees, Random Forest, Boosting, and XGBoost).

## Helpful Links
#### https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/
#### https://opendatascience.com/a-machine-learning-deep-dive-into-my-spotify-data/
#### https://www.youtube.com/watch?v=7J_qcttfnJA&t=384s



# Project Start
## EDA
# read in data, got from python using Spotipi with Spotify Web API
allData <- read.csv("~/Desktop/Spotify/featuresData.csv")
badData <- read.csv("~/Desktop/Spotify/badFeatures.csv")
goodData <- read.csv("~/Desktop/Spotify/goodFeatures.csv")

names(allData)
drop <- c("X", "id", "uri", "track_href", "analysis_url", "type")

allData = allData[,!(names(allData) %in% drop)]
badData = badData[,!(names(badData) %in% drop)]
goodData = goodData[,!(names(goodData) %in% drop)]

names(allData)


#Initial Summary of data
sumstat_fun<- function(x){
  c(Mean=mean(x, na.rm=T), Std=sd(x, na.rm = T), 
    Min=min(x, na.rm = T), Max=max(x, na.rm = T))
}

apply(badData, 2, sumstat_fun)
apply(goodData, 2, sumstat_fun)

par(mfrow=c(2,3))
boxplot(allData$danceability~allData$Like, names=c("Dislike", "Like"), ylab="Danceability", xlab="Rating")
boxplot(allData$energy~allData$Like, names=c("Dislike", "Like"), ylab="Energy", xlab="Rating")
boxplot(allData$loudness~allData$Like, names=c("Dislike", "Like"), ylab="Loudness", xlab="Rating")
boxplot(allData$acousticness~allData$Like, names=c("Dislike", "Like"), ylab="Acousticness", xlab="Rating")
boxplot(allData$valence~allData$Like, names=c("Dislike", "Like"), ylab="Valence", xlab="Rating")
boxplot(allData$key~allData$Like, names=c("Dislike", "Like"), ylab="Key", xlab="Rating")

par(mfrow=c(1,2))
boxplot(allData$speechiness~allData$Like, names=c("Dislike", "Like"), ylab="Speechiness", xlab="Rating")
boxplot(allData$liveness~allData$Like, names=c("Dislike", "Like"), ylab="Liveness", xlab="Rating")

par(mfrow=c(1,3))
boxplot(allData$tempo~allData$Like, names=c("Dislike", "Like"), ylab="Tempo", xlab="Rating")
boxplot(allData$duration_ms~allData$Like, names=c("Dislike", "Like"), ylab="Duration (ms)", xlab="Rating")
boxplot(allData$instrumentalness~allData$Like, names=c("Dislike", "Like"), ylab="Instrumentalness", xlab="Rating")


library(ROCR)
library(rpart)
library(rpart.plot)

set.seed(2020)
index<- sample(1:nrow(allData), 0.8*nrow(allData))
train <- allData[index,]
test <- allData[-index,]


### Basic Models
#### GLM with all variables
plainModel <- glm(Like~., data=train, family=binomial)
summary(plainModel)

reducedLM <- glm(Like~.-loudness -acousticness -time_signature -duration_ms -instrumentalness -tempo -speechiness -liveness, data=train, family=binomial)
summary(reducedLM)

#### GLM with reduced variables
nullmodel <- lm(Like~1, data=train)
fullmodel <- plainModel

# Forward
model.step.forward <- step(nullmodel, scope = list(lower=nullmodel, upper=fullmodel), direction = 'forward')
model.step.forwardBIC <- step(nullmodel, scope = list(lower=nullmodel, upper=fullmodel), direction = 'forward', k=log(nrow(train)))

# Backward
model.step.backward <- step(fullmodel, direction = 'backward')
model.step.backwardBIC <- step(fullmodel, direction = 'backward', k=log(nrow(train)))

# Stepwise
model.stepwise <- step(nullmodel, , scope = list(lower=nullmodel, upper=fullmodel), direction = 'both')
model.stepwiseBIC <- step(nullmodel, , scope = list(lower=nullmodel, upper=fullmodel), direction = 'both', k=log(nrow(train)))



## Tree Models
#### Tree with no set CP
plainTree <- rpart(formula = Like~., data=train, method="class")
prp(plainTree, digits = 4, extra = 1)

largeTree <- rpart(formula = Like~., data=train, method="class", cp=0.001)
prp(largeTree, digits = 4, extra = 1)
plotcp(largeTree)

##### Changing CP
setCPTree <- rpart(formula = Like~., data=train, method="class", cp=0.016)
prp(setCPTree, digits = 4, extra = 1)

largeTreeAsym <- rpart(formula = Like~., data=train, method = "class", parms = list(loss=matrix(c(0,1,10,0), nrow = 2)), cp=.001)
prp(largeTreeAsym, digits = 4, extra = 1)
plotcp(largeTreeAsym)


setCPTreeAsym <- rpart(formula = Like~., data=train, method="class", cp=0.0045)
prp(setCPTreeAsym, digits = 4, extra = 1)

##### Considering an asymmetric cost (weighted against false negatives)
largeTreeAsym <- rpart(formula = Like~., data=train, method = "class", parms = list(loss=matrix(c(0,1,10,0), nrow = 2)), cp=.001)
prp(largeTreeAsym, digits = 4, extra = 1)
plotcp(largeTreeAsym)


setCPTreeAsym <- rpart(formula = Like~., data=train, method="class", cp=0.0045)
prp(setCPTreeAsym, digits = 4, extra = 1)

## Finding optimal P Cut
### Function and calculation for optimal cutoff probability
pred.plainModel.train<- predict(plainModel, type="response")

# Function for weighting/weighted MR (cost)
costfunc = function(obs, pred.p, pcut){
  weight0 = 3    # weight for "true=0 but pred=1" -- FP
  weight1 = 1   # weight for "true=1 but pred=0" -- FN
  
  c0 = (obs==0)&(pred.p>=pcut)   # count where "true=0 but pred=1" -- FP
  c1 = (obs==1)&(pred.p<pcut)    # count where "true=1 but pred=0" -- FN
  
  cost = mean(weight1*c1 + weight0*c0)  # misclassification rate with weights applied
  return(cost) 
}

# Probability sequence for P-Cut, range is [0.1,1] and values are spread by .01
p.seq = seq(0.01, 1, 0.01) 

# Calculate cost (MR) for each probability in the p.seq sequence
cost = rep(0, length(p.seq))  # empty vector for storing calculated costs for each probability

for(i in 1:length(p.seq)){ 
  cost[i] = costfunc(obs = train$Like, pred.p = pred.plainModel.train, pcut = p.seq[i])  
  # uses cost function for each probability sequence; function returns cost which is then stored in the vector at point i
} 

# Assign new variable with the p-cut that has the lowest MR/Cost
pcut = p.seq[which(cost==min(cost))]  

# This is the naive pcut
expcut = mean(allData$Like)



## Using advanced models 
library(randomForest)
library(gbm)
library(parallel)
library(ipred)
library(rpart)
library(xgboost)

#### Bagging
train.Bag <- bagging(as.factor(Like)~., data = train, nbagg=100, 
                     method="class")       
summary(train.Bag)
predprob.test.bag<- predict(train.Bag, newdata = test, type="prob")[,2]        # This predicts the probabilistic prediction instead of classification. This is used to analyze/determine pcut values

pred.test.bag <- prediction(predprob.test.bag, test$Like)

perf.test.bag <- performance(pred.test.bag, "tpr", "fpr")
plot(perf.test.bag, colorize=TRUE)
#Get the AUC
train.Bag.AUC <- unlist(slot(performance(pred.test.bag, "auc"), "y.values"))

predclass.test.bag <- predict(train.Bag, newdata = test, type="class") 
train.Bag.ConfMatr <- table(test$Like, predclass.test.bag, dnn = c("True", "Pred"))             
train.Bag.RateTable <- prop.table(train.Bag.ConfMatr, 1)
mean(train.Bag.ConfMatr != test$Like)

classFunc.test <- (predprob.test.bag>pcut)*1
conf_matFunc <- table(test$Like, classFunc.test, dnn=c("True", "Prediction"))
rateTable <- prop.table(conf_matFunc, 1)
mean(classFunc.test != test$Like)


#### Random Forest
train.rf<- randomForest(as.factor(Like)~., data = train, cutoff=c(1/3, 2/3)) # classwt does not change the results; this cutoff would make the majority vote requirement for 0 to be 3/4 instead of .5 as before. This encourages a false positive over a false negative
train.rf


plot(train.rf, lwd=rep(2, 3))
legend("right", legend = c("OOB Error", "FPR", "FNR"), lwd=rep(2, 3), lty = c(1,2,3), col = c("black", "red", "green"))

# Prediction
train.rf.pred<- predict(train.rf, newdata=test, type = "prob")[,2]
rf.pred <- prediction(train.rf.pred, test$Like)
rf.perf <- performance(rf.pred, "tpr", "fpr")
plot(rf.perf, colorize=TRUE)
#Get the AUC
unlist(slot(performance(rf.pred, "auc"), "y.values"))   #AUC is pretty similar to bagging

rf.class.test<- predict(train.rf, newdata=test, type = "class")
rf.ConfMatr <- table(test$Like, rf.class.test, dnn = c("True", "Pred"))
prop.table(rf.ConfMatr, 1)
mean(rf.class.test != test$Like)

rf.class.test<- (train.rf.pred>pcut)*1      
rf.ConfMatre <- table(test$Like, rf.class.test, dnn = c("True", "Pred"))   
prop.table(rf.ConfMatre, 1)
mean(rf.class.test != test$Like)


classFunc.test <- (train.rf.pred>pcut)*1
conf_matFunc <- table(test$Like, classFunc.test, dnn=c("True", "Prediction"))
rateTable <- prop.table(conf_matFunc, 1)
mean(classFunc.test != test$Like)



#### Boosting
train.Boost <- gbm(Like~., data = train, distribution = "bernoulli", 
                   n.trees = 5000, cv.folds = 5, n.cores = 6)
summary(train.Boost)

best.iter <- (gbm.perf(train.Boost, method = "cv"))
best.iter

# Predict probability on test set
pred.train.boost<- predict(train.Boost, newdata = test, n.trees = best.iter, type="response")
# AUC
pred <- prediction(pred.train.boost, test$Like)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))

pred.train.boost.class<- (pred.train.boost>expcut)*1    # Classify based on prediction probability
boost.ConfMatre <- table(test$Like, pred.train.boost.class, dnn = c("True", "Pred"))
prop.table(boost.ConfMatre, 1)
mean(pred.train.boost.class != test$Like)



pred.train.boost.class<- (pred.train.boost>pcut)*1    # Classify based on prediction probability
boost.ConfMatre <- table(test$Like, pred.train.boost.class, dnn = c("True", "Pred"))  # Confusion Matrix
prop.table(boost.ConfMatre, 1)
mean(pred.train.boost.class != test$Like)


#### XGBoost
ptm <- proc.time()
fit.xgboost.class<- xgboost(data = model.matrix(~., train[,-14])[,-1], label = train$Like, 
                            eta = 0.1, nthread = 6, nrounds = 100, objective = "binary:logistic", verbose = 0)

pred.spotify.xgboost<- predict(fit.xgboost.class, newdata = model.matrix(~., test[,-14])[,-1])
# AUC
pred <- prediction(pred.spotify.xgboost, test$Like)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))

pred.spotify.xgboost.class<- (pred.spotify.xgboost>expcut)*1
xgBoostConf <- table(test$Like, pred.spotify.xgboost.class, dnn = c("True", "Pred"))
prop.table(xgBoostConf, 1)
mean(pred.spotify.xgboost.class != test$Like)


pred.spotify.xgboost.class<- (pred.spotify.xgboost>pcut)*1    # Classify based on prediction probability
xgBoostConf <- table(test$Like, pred.spotify.xgboost.class, dnn = c("True", "Pred"))  # Confusion Matrix
prop.table(xgBoostConf, 1)
mean(pred.spotify.xgboost.class != test$Like)



#### Comparing model diagnostics
################### 1. Obtain AIC, BIC for all models ################### 
compTable <- matrix(c(AIC(plainModel), BIC(plainModel), 
                      AIC(reducedLM), BIC(reducedLM), 
                      AIC(model.step.forward), BIC(model.step.forward), 
                      AIC(model.step.forwardBIC), BIC(model.step.forwardBIC),
                      AIC(model.step.backward), BIC(model.step.backward),
                      AIC( model.step.backwardBIC), BIC( model.step.backwardBIC),
                      AIC(model.stepwise), BIC(model.stepwise),
                      AIC(model.stepwiseBIC), BIC(model.stepwiseBIC)),
                    ncol=2, byrow=TRUE)
colnames(compTable) <- c("AIC", "BIC")
rownames(compTable) <- c("Full Model","Reduced Model", "Forward", "Forward BIC", "Backward", "Backward BIC", "Stepwise", "Stepwise BIC")

compTable




################### 2. Calculate AUC for all models on both training and testing samples   ################### 
### I used a function I wrote for this but I did test it on model0 and it returned the same values as when we solved for them in lab 2 (lines 60-85)

# Create Function
aucCalc <- function(model, TrainData, desTrainVar, TestData,desTestVar){
  # Training Set First
  modFit <- model$fitted.values
  ## training sample ROC curve and AUC
  pred <- prediction(modFit, desTrainVar)
  # gives AUC
  AUCvalTrain <- unlist(slot(performance(pred, "auc"), "y.values"))
  
  
  # Testing Set
  modFit <- predict(model, newdata = TestData, type="response")
  pred <- prediction(modFit, desTestVar)
  
  # gives AUC
  AUCvalTest <- unlist(slot(performance(pred, "auc"), "y.values"))
  
  
  AUCVals <- cbind(AUCvalTrain, AUCvalTest)
  return(AUCVals)
}

# Run function on all models, store values; only difference in these lines are the model being used in the function
modFullAUC <- aucCalc(model = plainModel, TrainData = train, desTrainVar = train$Like, TestData = test ,desTestVar = test$Like)  
modRedAUC <- aucCalc(model = reducedLM, TrainData = train, desTrainVar = train$Like, TestData = test ,desTestVar = test$Like)  
fAUC <- aucCalc(model = model.step.forward, TrainData = train, desTrainVar = train$Like, TestData = test ,desTestVar = test$Like)  
fBICAUC <- aucCalc(model = model.step.forwardBIC, TrainData = train, desTrainVar = train$Like, TestData = test ,desTestVar = test$Like)  
bAUC <- aucCalc(model = model.step.backward, TrainData = train, desTrainVar = train$Like, TestData = test ,desTestVar = test$Like)  
bBICAUC <- aucCalc(model = model.step.backwardBIC, TrainData = train, desTrainVar = train$Like, TestData = test ,desTestVar = test$Like)  
sAUC <- aucCalc(model = model.stepwise, TrainData = train, desTrainVar = train$Like, TestData = test ,desTestVar = test$Like)  
sBICAUC <- aucCalc(model = model.stepwiseBIC, TrainData = train, desTrainVar = train$Like, TestData = test ,desTestVar = test$Like)  


# Store all values from ^ in a table, print table
compTableAUC <- matrix(c(modFullAUC,
                         modRedAUC,
                         fAUC, 
                         fBICAUC,
                         bAUC,
                         bBICAUC,
                         sAUC,
                         sBICAUC),
                       ncol=2, byrow=TRUE)
colnames(compTableAUC) <- c("Train AUC", "Test AUC")
rownames(compTableAUC) <- c("Full Model","Reduced Model", "Forward", "Forward BIC", "Backward", "Backward BIC", "Stepwise", "Stepwise BIC")

compTableAUC


################### 3. What are MR, FPR, FNR for all models on testing sample? ################### 

# Function Definition
predAndErr <- function(model, testdata, pcut, desTestVar){
  predictions <- predict(model, newdata = testdata, type="response")
  classFunc.test <- (predictions>pcut)*1
  conf_matFunc <- table(desTestVar, classFunc.test, dnn=c("True", "Prediction"))
  rateTable <- prop.table(conf_matFunc, 1)
  
  MR <- mean(classFunc.test != desTestVar)
  FPR <- rateTable[1,2]
  FNR <- rateTable[2,1]
  
  results <- cbind(MR, FPR, FNR)
  return(results)
}

# Run Function on all models, save results
modFullPred <- predAndErr(model=plainModel, testdata=test, pcut=pcut, test$Like)
modRedPred <- predAndErr(model=reducedLM, testdata=test, pcut=pcut, test$Like)
modFPred <- predAndErr(model=model.step.forward, testdata=test, pcut=pcut, test$Like)
modFBPred <- predAndErr(model=model.step.forwardBIC, testdata=test, pcut=pcut, test$Like)
modBPred <- predAndErr(model=model.step.backward, testdata=test, pcut=pcut, test$Like)
modBBPred <- predAndErr(model=model.step.backwardBIC, testdata=test, pcut=pcut, test$Like)
modSPred <- predAndErr(model=model.stepwise, testdata=test, pcut=pcut, test$Like)
modSBPred <- predAndErr(model=model.stepwiseBIC, testdata=test, pcut=pcut, test$Like)


# Store all values from ^ in a table, print table
compTableMRErr <- matrix(c(modFullPred, 
                           modRedPred,
                           modFPred, 
                           modFBPred,
                           modBPred,
                           modBBPred,
                           modSPred,
                           modSBPred),
                         ncol=3, byrow=TRUE)
colnames(compTableMRErr) <- c("MR", "FPR", "FNR")
rownames(compTableMRErr) <- c("Full Model","Reduced Model", "Forward", "Forward BIC", "Backward", "Backward BIC", "Stepwise", "Stepwise BIC")

compTableMRErr

################################################################## Exercise Results Only ################################################################## 
cbind(compTable, compTableAUC, compTableMRErr)