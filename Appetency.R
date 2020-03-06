#----------------------------------------------------------------------------#

#Team Leader: Rubaida Easmin
#Nidya Esperanza Ballesteros Avila
#Gozde Orhan
#9/04/2019

#----------------------------------------------------------------------------#

#remove workspace
rm(list=ls())

#--------------------------Libraries----------------------------------------#
library(xgboost)
library(klaR)
#library(klaR, lib.loc="/home/gorha001/my_RM_work")
library(e1071)
library(glmnet)
library(caret)
library(corrplot)
library(reshape2)
library(dplyr)
library(DMwR)
library(rJava)
library(FSelector)
library(CORElearn)
library(AppliedPredictiveModeling)
library(randomForest)
library(funModeling)
#library(funModeling, lib.loc="/home/gorha001/my_RM_work")
library(rpart)
library(leaps)
library(MASS)
library(nnet)
library(RANN)
#library(RANN, lib.loc="/home/gorha001/my_RM_work")


#setwd('/Users/gozdeorhan/Desktop/ML_CW2/Esperanza')
#setwd('/home/gorha001/my_RM_work')

#---------------------------------Datasets--------------------------------------#

train_x=read.delim("train_X.csv", stringsAsFactors = FALSE, header=FALSE,na.strings=c(""," ",NA))
dim(train_x)

train_y=read.delim("train_Y.csv", stringsAsFactors = FALSE, header=TRUE,na.strings=c(""," ",NA))
dim(train_y)

test_x=read.delim("test_X.csv", stringsAsFactors = FALSE, header=FALSE,na.strings=c(""," ",NA))
dim(test_x)

#------------------Removing rows and columns with >%70 NAs----------------------#

#no of missing values column wise
miss_val_col = colSums(is.na(train_x))

#remove column
train_x = train_x[,colMeans(is.na(train_x)) <= 0.7]
dim(train_x)

#no of missing values row wise
miss_val_row = rowSums(is.na(train_x))

#remove rows
train_x = train_x[rowMeans(is.na(train_x)) <= 0.7,]
dim(train_x)

#----------Removing columns with zero variance and near zero variance----------#

#check near zero and zero variance
predictorInfo = nearZeroVar(train_x,saveMetrics = TRUE)

#column names that have zero variance
rownames(predictorInfo)[predictorInfo$zeroVar]

#column names that have near zero variance
rownames(predictorInfo)[predictorInfo$nzv]

#remove nzv column from dataset
train_x = train_x[,!predictorInfo$nzv]

#-----Manipulate the columns of test data to have same columns with train set------#

testkeep1=colnames(train_x)
test_x = test_x[testkeep1]
dim(test_x)

#------------------Combine train data with target value------------------------#

dfa = cbind(train_x,train_y['appetency'])
dfa$appetency = as.factor(dfa$appetency)
dim(dfa)
summary(dfa)


#-------Create a new dataframe to be used to check informative missingness-------#

dfa.NA = dfa
for (i in colnames(dfa.NA)){                             #convert NA to 1/0
  if (sum(is.na(dfa.NA[,i]))>0) {
    dfa.NA[,paste(i,"NA",sep="")] = ifelse(is.na(dfa.NA[[i]]), "1", "0")
    dfa.NA = dfa.NA[,!names(dfa.NA) %in% i]
  }
}
dim(dfa.NA)

#----------------------------------------------------------------------------#

test.NA = test_x
for (i in colnames(test.NA)){                             #convert NA to 1/0
  if (sum(is.na(test.NA[,i]))>0) {
    test.NA[,paste(i,"NA",sep="")] = ifelse(is.na(test.NA[[i]]), "1", "0")
    test.NA = test.NA[,!names(test.NA) %in% i]
  }
}
dim(test.NA)

#--------Decision tree to Decision tree to check informative missingness-----------------#

#dfa.NA.dtree=rpart(appetency~.,data=dfa.NA,method="class",control=rpart.control(minsplit=30,cp=0.001))
#varImp(dfa.NA.dtree) # important columns V70NA and V84NA 


#Adding V70NA and V84NA columns to the dfa.train dataset
dfa$V36NA=dfa.NA$V70NA
dfa$V4NA=dfa.NA$V84NA

#add V36NA and V4NA with the test dataset
test_x$V70NA=test.NA$V70NA
test_x$V84NA=test.NA$V84NA


dfa[sapply(dfa, is.character)] = lapply(dfa[sapply(dfa, is.character)],as.factor) #convert char to factor
test_x[sapply(test_x, is.character)] = lapply(test_x[sapply(test_x, is.character)],as.factor) #convert char to factor

#------------Split the data-----------------------------------------------------------------------------------------------------------

set.seed(234)
dfa<-dfa[sample(nrow(dfa)),] # Randomly shuffle the data

split_a <-createDataPartition(dfa$appetency, p = 0.70)[[1]] #Splitting the data into training and testing
dfa.train<-dfa[split_a,]
dfa.validation<-dfa[-split_a,]
dim(dfa.train)
dim(dfa.validation)

#------------------Check balance of dataset------------------------------------------------------------------------------------

barplot(prop.table(table(dfa.train$appetency)),xlab = "Appetency distribution",ylab = "Frequency of the sample")

#-------------------Correlation calculation------------------------------------------------------------------------#

numeric.dfa = dfa.train[sapply(dfa.train, is.numeric)] # find numeric train set
#calculate correlation on numeric values
correlation_dfa = cor(numeric.dfa,use='pairwise.complete.obs')
corrplot.mixed(correlation_dfa) # plot correlation matrix

#column names that are above 75%  threshold
corr_colnames = findCorrelation(correlation_dfa,cutoff = 0.75,names = TRUE) #9 columns

#remove those columns
dfa.train = dfa.train[, !(names(dfa.train) %in% corr_colnames)]
dfa.validation = dfa.validation[, !(names(dfa.validation) %in% corr_colnames)]
test_x = test_x[, !(names(test_x) %in% corr_colnames)]
dim(test_x)
dim(dfa.validation)
dim(dfa.train)

#--------------------------Imputation (Train)-----------------------------------
#--create dataframe with numerical column
num.dfa.train = dfa.train[sapply(dfa.train, is.numeric)]
#-numerical imputation with KNN 
dfa.train.imp_n = preProcess(dfa.train, method = c("knnImpute","center", "scale"))
dfa.train.pred.num = predict(dfa.train.imp_n, num.dfa.train)
#dfa.train.pred.num 
colSums(is.na(dfa.train.pred.num))

#-#create dataframe with categorical column
categorical.dfa.train = dfa.train[sapply(dfa.train, is.factor)]
# KNN IMPUTATION for categorical values
#dfa.train.imp_c = knnImputation(categorical.dfa.train, k=6)
#colSums(is.na(dfa.train.imp_c))
#write.csv(dfa.train.imp_c,file = "knn6_cat_dfa.train.csv",row.names = FALSE)#written in a file as it take too much time

dfa.train.imp_c = read.csv(file="knn6_cat_dfa.train.csv", header=TRUE) #read the imputed file

dfa.train.imp_c = dfa.train.imp_c[c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30,31,29)]

#-convert appetency, v70NA and V84NA from int to factor----
dfa.train.imp_c$appetency = as.factor(dfa.train.imp_c$appetency)
dfa.train.imp_c$V70NA = as.factor(dfa.train.imp_c$V70NA)
dfa.train.imp_c$V84NA = as.factor(dfa.train.imp_c$V84NA)

#To create the final data train
dfa.train.imputed = cbind(dfa.train.pred.num,dfa.train.imp_c)
dim(dfa.train.imputed) #58

#-----------------------Imputation (Validation set)--------------------------------------------------------
num.dfa.val = dfa.validation[sapply(dfa.validation, is.numeric)]
#-numerical
dfa.val.pred.num = predict(dfa.train.imp_n, num.dfa.val) ####use the imputation from the training set
colSums(is.na(dfa.val.pred.num))

#-create dataframe with categorical column
categorical.dfa.val = dfa.validation[sapply(dfa.validation, is.factor)]
#dfa.val.imp_c = knnImputation(categorical.dfa.val, k=6)
#colSums(is.na(dfa.val.imp_c))
#write.csv(dfa.val.imp_c,file = "knn6_cat_dfa.val.csv",row.names = FALSE)#written in a file as it take too much time

dfa.val.imp_c = read.csv(file="knn6_cat_dfa.val.csv", header=TRUE) #read the imputed file

dfa.val.imp_c = dfa.val.imp_c[c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30,31,29)]

#-convert appetency, v70NA and V84NA from int to factor--
dfa.val.imp_c$appetency = as.factor(dfa.val.imp_c$appetency)
dfa.val.imp_c$V70NA = as.factor(dfa.val.imp_c$V70NA)
dfa.val.imp_c$V84NA = as.factor(dfa.val.imp_c$V84NA)

#-Combine the imputation for numerical & categorical colums----
#To create the final data train
dfa.val.imputed = cbind(dfa.val.pred.num, dfa.val.imp_c)
dim(dfa.val.imputed) #58

#------------------------------imputation(test)---------------------------#

num.test = test_x[sapply(test_x, is.numeric)]
test_x.imput.n = predict(dfa.train.imp_n,num.test)

#create dataframe with categorical column
categorical.test_a = test_x[sapply(test_x, is.factor)]

#test.imp_c = knnImputation(categorical.test_a) ##KNN IMPUTATION for categorical values
#write.csv(test.imp_c,file = "test.knn.dfa.categ.csv",row.names = FALSE) ##write in a file as it take too much time

test.imp_c = read.csv(file="test.knn.dfa.categ.csv", header=TRUE) #read the imputed file

test.imp_c$V70NA = as.factor(test.imp_c$V70NA)
test.imp_c$V84NA = as.factor(test.imp_c$V84NA)

test.imputed = cbind(test_x.imput.n,test.imp_c)  ## combine both dataframe to create final train data
dim(test.imputed)
colSums(is.na(test.imputed))

#------------------------Balancing the data (Train)-----------------------------

as.data.frame(table(dfa.train.imputed$appetency)) 
dfa.train.balance = SMOTE(appetency~., dfa.train.imputed, perc.over = 2800, k=5, perc.under = 150)       #balance trainset
as.data.frame(table(dfa.train.balance$appetency))                              #check the balance of target column
dfa.train.f = dfa.train.balance
dim(dfa.train.f)

#----------------Feature Selection (RELIEF)--------------------------------

#relief.dfa<-attrEval("appetency", 
#                     data=dfa.train.f,
#                     estimator="ReliefFequalK",
#                     ReliefIterations=50)
#head(relief.dfa)
#relief.dfa.col = names(sort(abs(relief.dfa), decreasing = TRUE)[1:40])
#relief.dfa.col

#------------------Feature Selection (RELIEF/Permutation)------------------------------------------------------#
#check importance of categorical value(balanced data) (RELIEF/Information Gain Ratio) relief scores

#perm.dfa<-permuteRelief(x=dfa.train.f[,-ncol(dfa.train.f)],y=dfa.train.f[,ncol(dfa.train.f)],
                        #nperm = 500, estimator = "ReliefFequalK",
                        #ReliefIterations = 50)

#head(perm.dfa$permutations)
#str(perm.dfa$permutations)
#histogram(~value|Predictor,
          #data=perm.dfa$permutations)

#permtation singficant variables
#pred.90.col.dfa<-names(sort(abs(perm.dfa$standardized[which(abs(perm.dfa$standardized)>=1.6)]), decreasing=T))
#print(pred.90.col.dfa) # 38 columns

#----------------------Feature Selection(ROC-filterVarImp)-----------------------------
#check importance of categorical value(balanced data) (ROC Curve)

#rocVal.dfa = filterVarImp(filterVarImp(x=dfa.train.f[,-ncol(dfa.train.f)],
                                       #y=dfa.train.f[,ncol(dfa.train.f)])

#rocVal.order.dfa = order( rocVal.dfa$X.1, decreasing=TRUE )     # Sort by the value of the variable importance:
#for (rn in 1:length(rocVal.order.dfa)) {
#  print(rownames(rocVal.dfa)[rocVal.order.dfa[rn]])
#}


#-------------------------Feature Selection(information gain based on entropy)---------------

#info.score = information.gain(appetency~., dfa.train.f)
#subset=cutoff.k(info.score, 2)
#f=as.simple.formula(info.score$attr_importance, "appetency")
#print(f)
#infoScore.order = order( info.score$attr_importance, decreasing=TRUE )  # Sort by the value of the variable importance:
#for (rn in 1:length(infoScore.order)) {
#  print(rownames(info.score)[infoScore.order[rn]])
#}

#---------------------Select columns based on attrEval, permRelief, filterVarImp, information gain (Train)--------------------------
#selected 34 variables
#keep those columns selected by attrEval, permRelief, filterVarImp and information gain 
#Create a new data train with the variables which where commun on the four feature selecion 34 variables
keeps.dfa = c("appetency","V93","V165","V187","V138","V119","V116","V90","V84","V118","V155","V190",
              "V224","V36","V67","V126","V42","V204","V4","V114","V154","V189","V115","V212","V95","V29",
              "V149","V112","V107","V111","V192","V156","V14","V101")


dfa.train_updated = dfa.train.f[keeps.dfa]
dim(dfa.train_updated)
colSums(is.na(dfa.train_updated)) #check for missing value

#--------------------Select columns based on attrEval, permRelief, filterVarImp (Validation)-------------#
dfa.val.updated = dfa.val.imputed[keeps.dfa]
dim(dfa.val.imputed)
colSums(is.na(dfa.val.imputed))

#-------Select columns based on previous feature selection methods (Test)---------#
testkeep2=c("V93","V165","V187","V138","V119","V116","V90","V84","V118","V155","V190",
            "V224","V36","V67","V126","V42","V204","V4","V114","V154","V189","V115","V212","V95","V29",
            "V149","V112","V107","V111","V192","V156","V14","V101")
test.imputed = test.imputed[testkeep2]
dim(test.imputed)
colSums(is.na(test.imputed)) 

#-------Chi-square (Train)---------#
#(categorical data)
categorical.dat = dfa.train_updated[sapply(dfa.train_updated, is.factor)] # find categorical dataset
numerical.dat = dfa.train_updated[sapply(dfa.train_updated, is.numeric)]

chi.res = chi.squared(appetency~.,categorical.dat)
sort.chi = order(chi.res$attr_importance,decreasing = TRUE)
for (rn in 1:length(sort.chi)) {
  print(rownames(chi.res)[sort.chi[rn]])
}


#--------------------minimize labels in categorical columns with auto_grouping (Train)--------------------------------------------------------
#(categorical data)
categorical.dat = dfa.train_updated[sapply(dfa.train_updated, is.factor)] # find categorical dataset
numerical.dat = dfa.train_updated[sapply(dfa.train_updated, is.numeric)]

newDF=categorical.dat
testDF = categorical.dat
columnList = NULL
index = 1
for (col in 2:length(colnames(categorical.dat))) {
  col.levels = nlevels(categorical.dat[[colnames(categorical.dat)[col]]])
  #print(col.levels)
  if(col.levels>100){
    print(colnames(categorical.dat)[col])
    columnName = colnames(categorical.dat)[col]
    columnList [[index]] = columnName
    x= auto_grouping(data = categorical.dat, input = colnames(categorical.dat)[col], target = 'appetency',n_groups = 10,model = "kmeans",seed = 999)
    #print(x$recateg_results)
    newDF = merge(newDF, x$df_equivalence, by = colnames(categorical.dat)[col])
    testDF = merge(testDF, x$df_equivalence, by = colnames(categorical.dat)[col])
    head(testDF)
    newDF[columnName] = NULL
    index = index+1
  }
  testDF = testDF
}

#--------------------------------Combine numerical and categorical data-----------------------------------------
dfa.train_updated = cbind(newDF,numerical.dat)
dim(dfa.train_updated)
dfa.train_updated[sapply(dfa.train_updated, is.character)] = lapply(dfa.train_updated[sapply(dfa.train_updated, is.character)],as.factor)
dim(dfa.train_updated)
colSums(is.na(dfa.train_updated)) #ok

#--------------------minimize labels in categorical columns(Validation)-------------------#
categorical_val.dat = dfa.val.updated[sapply(dfa.val.updated, is.factor)]         #find categorical dataset
numerical_val.dat = dfa.val.updated[sapply(dfa.val.updated, is.numeric)]         #find numerical dataset

newDF.Val = categorical_val.dat

for (col in 1:length(columnList)) {
  print(columnList[col])
  y = auto_grouping(data = categorical_val.dat, input = columnList[col], target = 'appetency',n_groups = 10,model = "kmeans",seed = 999)
  #print(y$recateg_results)
  newDF.Val = merge(newDF.Val, y$df_equivalence, by = columnList[col])
  newDF.Val[columnList[col]] = NULL
}

dfa.validation_updated = cbind(newDF.Val,numerical_val.dat)


dfa.validation_updated[sapply(dfa.validation_updated,is.character)] = lapply(dfa.validation_updated[sapply(dfa.validation_updated, is.character)],as.factor)
dim(dfa.validation_updated)
colSums(is.na(dfa.validation_updated)) 

#-----------------------minimize labels in categorical columns(Test)-----------------------------------------#

cat.test = test.imputed[sapply(test.imputed, is.factor)]         #find categorical dataset
numerical.test = test.imputed[sapply(test.imputed, is.numeric)]  #find numerical dataset

newDF.test = cat.test

for (i in 1:length(columnList)) { 
  original.column = columnList[i]
  columnN = paste(original.column,"_rec",sep = "")
  collist = testDF[[columnN]]
  names(collist) = testDF[[original.column]]
  print(original.column)
  for (value in 1:length(cat.test[[original.column]])) {                  
    replaceCol = as.character(cat.test[[original.column]][value])
    print(replaceCol)
    newDF.test[[columnN]][value] = collist[replaceCol]
    newDF.test[[original.column]] = NULL
    print(collist[replaceCol])
  }
}

newDF.test[sapply(newDF.test,is.character)] = lapply(newDF.test[sapply(newDF.test, is.character)],as.factor)

dim(newDF.test)
colSums(is.na(newDF.test))
colSums(is.na(numerical.test))

#------------------------------imputation(test)---------------------------#

#newDF.test.imp_c = knnImputation(newDF.test) ##KNN IMPUTATION for categorical values
#write.csv(newDF.test.imp_c,file = "test.knn3_a.csv",row.names = FALSE) ##write in a file as it take too much time

test.imp_c = read.csv(file="test.knn3_a.csv", header=TRUE) #read the imputed file
test_x_updated = cbind(test.imp_c,numerical.test)
dim(test_x_updated)
(colSums(is.na(test_x_updated)))


#--------------------------------Dummification(Train)-----------------------------#
#creating dummy variables for factor type variables
dfa.dummies = dummyVars( ~., data = dfa.train_updated[,-1],levelsOnly = FALSE)
data_dummy=predict(dfa.dummies,newdata= dfa.train_updated)
data_dummy=as.data.frame(data_dummy)
data_dummy=data.frame(dfa.train_updated$appetency,data_dummy)
names(data_dummy )[1]="appetency"
dfa.train = data_dummy 
dim(dfa.train) #443 variables

#--------------------------------Dummification(Validation)-----------------------------#
#creating dummy variables for factor type variables
levels(dfa.validation_updated$V190) = levels(dfa.train_updated$V190)
levels(dfa.validation_updated$V204) = levels(dfa.train_updated$V204)
levels(dfa.validation_updated$V154) = levels(dfa.train_updated$V154)
levels(dfa.validation_updated$V29) = levels(dfa.train_updated$V29)


data_dummyVa=predict(dfa.dummies,newdata= dfa.validation_updated)
data_dummyVa=as.data.frame(data_dummyVa)
data_dummyVa=data.frame(dfa.validation_updated$appetency,data_dummyVa)
names(data_dummyVa)[1]="appetency"
dfa.validation = data_dummyVa 
dim(dfa.validation) #443 variables
(colSums(is.na(dfa.validation_updated)))

#---------------------------Dummification (Test)-----------------------------#

test_x_updated<-test_x_updated[!(test_x_updated$V29=="_URKn_77G3QyQHSVQ2N1RKmtW"),]
test_x_updated<-droplevels(test_x_updated, exclude = if(anyNA(levels(test_x_updated))) NULL else NA)

levels(test_x_updated$V190) = levels(dfa.train_updated$V190)
levels(test_x_updated$V204) = levels(dfa.train_updated$V204)
levels(test_x_updated$V126_rec) = levels(dfa.train_updated$V126_rec)
levels(test_x_updated$V4_rec) = levels(dfa.train_updated$V4_rec)
levels(test_x_updated$V29) = levels(dfa.train_updated$V29)
levels(test_x_updated$V154) = levels(dfa.train_updated$V154)

data_dummyT=predict(dfa.dummies,newdata= test_x_updated)
data_dummyT=as.data.frame(data_dummyT)
dfa.test = data_dummyT 
dim(dfa.test)
sum(is.na(dfa.test))

#--------------------------#FEATURE SELECTION-------------------------------------#
#--------------------------LASSO Regression----------------------------------#
library(glmnet)
set.seed(123)
x = model.matrix(appetency~., dfa.train)[,-1]
y = ifelse(dfa.train$appetency == "1", 1, 0)
cv.lasso = cv.glmnet(x, y, alpha = 1, family = "binomial")
plot(cv.lasso)
print(cv.lasso$lambda.min)
print(coef(cv.lasso, cv.lasso$lambda.min))
lasso.col = coef(cv.lasso, cv.lasso$lambda.min)
#write.table(as.data.frame.matrix(lasso.col), file = "lasso_dfa.train2a.csv",row.names=FALSE,sep = ",",col.names = FALSE)

#update data frame after lasso (remove 201 columns)
c=2
keep_lassoCol = NULL
l.xlsx = read.delim("lasso_dfa.train2a.csv",sep = ",",header = FALSE,stringsAsFactors = FALSE)
for (i in 1:length(l.xlsx[[1]])) {
  if (l.xlsx[[1]][i]!=0) {
    print(l.xlsx[[2]][i])
    keep_lassoCol[c] = l.xlsx[[2]][i]
    c=c+1
  }  
}


keep_lassoCol[1] = "appetency"
dfa.train = dfa.train[keep_lassoCol]
dim(dfa.train) # now we have 242 variables
(colSums(is.na(dfa.train)))

#--
dfa.validation = dfa.validation[keep_lassoCol]
dim(dfa.validation)  

dfa.test = dfa.test[keep_lassoCol[-1]]
dim(dfa.test)

####-------------------
#---------------------Use VIF for multicolinearity--------------------------#
#install.packages("DAAG") 
library(DAAG)
set.seed(123)
dfa.logitm = glm(appetency~.,data = dfa.train, family = binomial) # CREATE LOGISTIC MODEL WITH ALL VARIABLES
vif.value = DAAG::vif(dfa.logitm)
vif.value

##remove columns over to  5

remove_col = c("V29.F2FyR07IdsN7I",'V93_rec.group_3')

dfa.train = dfa.train[, !(colnames(dfa.train) %in% remove_col)]
dim(dfa.train)

dfa.validation = dfa.validation[, !(colnames(dfa.validation) %in% remove_col)]
dim(dfa.validation)

dfa.test = dfa.test[, !(colnames(dfa.test) %in% remove_col)]
dim(dfa.test) 

#--------------------Feature Selection with C5.0--------------------------------------#
install.packages("C50")
library(C50)
dfa.model_c5 = C5.0(appetency ~ ., data=dfa.train,trials=50,
                control = C5.0Control(                        
                  noGlobalPruning = T,
                  CF=0.8,
                  minCases=10,
                  sample = 0.80,
                  winnow=F,
                  earlyStopping=T
                ))

C5imp(dfa.model_c5)

#revove the zero ones
c=2
dfa.keep_c5Col = NULL
dfa.c5.col = read.delim("dfaC5.csv",sep = " ",header = FALSE,stringsAsFactors = FALSE)
for (i in 1:length(dfa.c5.col[[2]])) {
  if (dfa.c5.col[[2]][i]!=0) {
    print(dfa.c5.col[[1]][i])
    dfa.keep_c5Col[c] = dfa.c5.col[[1]][i]
    print(c)
    c=c+1
  }  
}

#REMOVE 7 COLUMNS
dfa.keep_c5Col[1] = "appetency"
dfa.train = dfa.train[dfa.keep_c5Col]
dim(dfa.train) # 233 variables

dfa.validation = dfa.validation[dfa.keep_c5Col]
dim(dfa.validation)  # 233 variables

dfa.test = dfa.test[dfa.keep_c5Col[-1]]
dim(dfa.test)

#--------------------------#FEATURE Selection with P-value-------------------------------------#
#set.seed(123)
#dfa.logit = glm(appetency~.,data = dfa.train, family = binomial) # CREATE LOGISTIC MODEL WITH ALL VARIABLES
#summary(dfa.logit)

#dfa.pv.imp = data.frame(summary(dfa.logit)$coef[summary(dfa.logit)$coef[,4] <= .05, 4])
#write.table(dfa.pv.imp, file = "dfa.logit_pv2a.csv",row.names=TRUE,sep = ",",col.names = TRUE)

c=2
dfa.keep_pvalueCol = NULL
dfa.pValue.col = read.delim("dfa.logit_pv2a.csv",sep = ",",header = FALSE,stringsAsFactors = FALSE)
for (i in 3:length(dfa.pValue.col[[1]])) {
  dfa.keep_pvalueCol[c] = dfa.pValue.col[[1]][i]
  c=c+1
}

dfa.keep_pvalueCol[1] = "appetency"
dfa.train = dfa.train[dfa.keep_pvalueCol]
dim(dfa.train) #67 variables
(colSums(is.na(dfa.train)))

dfa.validation = dfa.validation[dfa.keep_pvalueCol]
dim(dfa.validation) #67 variables
(colSums(is.na(dfa.validation))) # No NA's

dfa.test = dfa.test[dfa.keep_pvalueCol[-1]]
dim(dfa.test)
(colSums(is.na(dfa.test)))

#----------------------Feature Selection with Random Forest--------------------------------------#

dfa.train.rf = randomForest(appetency ~ ., data=dfa.train, mtry=15,
                            importance=TRUE, na.action=na.omit)
varImpPlot(dfa.train.rf)
dfa_rf.feature = dfa.train.rf$importance
#write.table(as.data.frame(dfa_rf.feature), file = "dfa.rf_fs.csv",row.names=TRUE,sep = ",",col.names = TRUE)


#----------------------------Feature Selection with RFE----------------------------------------------#

dfa.RFE.control = rfeControl(functions=ldaFuncs, method="cv", number=3)
dfa.RFE.results = rfe(dfa.train[,2:67], dfa.train[,1], sizes=c(2:67), rfeControl=dfa.RFE.control)
#summarize the results
print(dfa.RFE.results)
#list the chosen features
predictors(dfa.RFE.results)


#--------------------------Columns after RFE and Random Forest(50)-----------------------------------------#
c=1
keep = NULL
columns.xlsx = read.delim("a_columns.csv",sep = ",",header = FALSE,stringsAsFactors = FALSE)
for (i in 1:length(columns.xlsx[[1]])) {
  keep[c] = columns.xlsx[[1]][i] #50 columns
  c=c+1
}  

dfa.train = dfa.train[keep]
dim(dfa.train) #50 columns

dfa.validation = dfa.validation[keep]
dim(dfa.validation)  #50 columns

dfa.test = dfa.test[keep[-1]]
dim(dfa.test)  #49 columns

#--------------------------MODEL FITTING-------------------------------------#
#--------------------------MODELS------------------------------#

#Function for measuring AUC
get_auc <- function(tab) {
  tot = colSums(tab)
  truepos = unname(rev.default(cumsum(rev.default(tab[2,]))))
  falsepos = unname(rev.default(cumsum(rev.default(tab[1,]))))
  totpos = sum(tab[2,])
  totneg = sum(tab[1,])
  sens = truepos/totpos
  omspec = falsepos/totneg
  sens = c(sens,0)
  omspec = c(omspec,0)
  height = (sens[-1]+sens[-length(sens)])/2
  width = -diff(omspec)
  auc = sum(height*width)
  return(auc)
}

ctrl = trainControl(method = "cv", number = 10)
#------------------Decision Tree-----------------------#

set.seed(123)
dfa.tree_fit = train(appetency ~., data = dfa.train, method = "rpart",
                     preProcess=c("center", "scale","BoxCox"),
                     trControl=ctrl,
                     tuneLength = 10)

#Prediction on validation dataset
dfa.test_pred = predict(dfa.tree_fit, newdata = dfa.validation,na.action = na.pass)
confusionMatrix(dfa.test_pred, dfa.validation$appetency,positive="1")

#Prediction on test dataset
dta_pred2 = predict(dfa.tree_fit, newdata = dfa.test,na.action = na.pass)
write.csv(dta_pred2,file = "appetency_prediction.csv",row.names = FALSE)

tab = as.matrix(table(dfa.test_pred,dfa.validation$appetency))
get_auc(tab)

#-------------------Random Forest-----------------------#
set.seed(123)

#TRAIN MODEL
dfa.tunegrid=expand.grid(.mtry=7)

dfa.rf_fit = train(appetency~., data=dfa.train, 
                   method='rf',  
                   metric='Accuracy', 
                   preProcess=c("center", "scale","BoxCox"),
                   tuneGrid=dfa.tunegrid, 
                   trControl=ctrl)

#Prediction on validation dataset
dfa.rf.predict=predict(dfa.rf_fit,newdata = dfa.validation,na.action = na.pass)
confusionMatrix(dfa.rf.predict, dfa.validation$appetency,positive='1')


#Prediction on test dataset
rf_pred2 = predict(dfa.rf_fit, newdata = dfa.test,na.action = na.pass)
rf_pred2

tab_rf = as.matrix(table(dfa.rf.predict,dfa.validation$appetency))
get_auc(tab_rf)

#-------------------------Bagging----------------------------------#
set.seed(123)

dfatunegrid = expand.grid(.mtry=9)
dfa.bag_fit = train(appetency~., 
                    data=dfa.train,
                    method='treebag', 
                    metric='Accuracy',
                    preProcess=c("center", "scale","BoxCox","spatialSign"),
                    tunegrid = dfatunegrid,
                    trControl=ctrl)

#Prediction on validation dataset
dfa.bag.predict = predict(dfa.bag_fit, newdata = dfa.validation)
confusionMatrix(dfa.bag.predict, dfa.validation$appetency,positive = '1')

#Prediction on test dataset
bag_pred2 = predict(dfa.bag_fit, newdata = dfa.test,na.action = na.pass)
bag_pred2

tab_bag = as.matrix(table(dfa.bag.predict,dfa.validation$appetency))
get_auc(tab_bag)

#----------------------------SVM------------------------------------#
set.seed(123)

dfa.SVModel = train(appetency ~ ., data = dfa.train,
                method = "svmPoly",
                trControl= ctrl,
                tuneGrid = data.frame(degree = 1,
                                      scale = 1,
                                      C = 1),
                preProcess = c("scale","center"),
                na.action = na.omit)


#Prediction on validation dataset
dfa.svmPred<-predict(dfa.SVModel,dfa.validation[,names(dfa.validation)!="appetency"])
confusionMatrix(dfa.svmPred,dfa.validation$appetency,positive='1')

#Prediction on test dataset
svm_pred2 = predict(dfa.SVModel, newdata = dfa.test,na.action = na.pass)
svm_pred2

tab_svm = as.matrix(table(dfa.svmPred,dfa.validation$appetency))
get_auc(tab_svm)

#----------------------------KNN------------------------------------#
set.seed(123)

dfa.knn_fit = train(dfa.train[,-1], dfa.train$appetency, method = "knn",
                    trControl=ctrl,
                    preProcess = c("center", "scale","BoxCox","spatialSign"),
                    tuneLength = 10)

#Prediction on validation dataset
dfa.knn_pred = predict(dfa.knn_fit, newdata = dfa.validation)
confusionMatrix(dfa.knn_pred,dfa.validation$appetency,positive = '1')

#Prediction on test dataset
knn_pred2 = predict(dfa.knn_fit, newdata = dfa.test,na.action = na.pass)
knn_pred2

tab_knn = as.matrix(table(dfa.knn_pred,dfa.validation$appetency))
get_auc(tab_knn)

#----------------------Naive Bayes Algorithm--------------------------#

set.seed(123)
dfa.nb_fit = train(dfa.train[,-1], dfa.train$appetency, method="nb", 
               preProcess=c("center", "scale","BoxCox","spatialSign"),
               trControl=ctrl)

#Prediction on validation dataset
dfa.nb_pred = predict(dfa.nb_fit, dfa.validation)
confusionMatrix(dfa.nb_pred, dfa.validation$appetency, positive="1")

#Prediction on test dataset
nb_pred2 = predict(dfa.nb_fit, newdata = dfa.test,na.action = na.pass)
nb_pred2

tab_nb = as.matrix(table(dfa.nb_pred,dfa.validation$appetency))
get_auc(tab_nb)

#----------------------Logistic Regression-------------------------------------------#
set.seed(123)
logit = glm(appetency~.,data=dfa.train,family = binomial)
glm.probs = predict(logit, dfa.validation, type='response')
glm.pred <- ifelse(glm.probs>0.5,'-1','1')
mean(glm.pred==dfa.validation$appetency)
threshold <- 0.5
confusionMatrix(factor(glm.pred),factor(dfa.validation$appetency),positive="1")

tab_log = as.matrix(table(factor(glm.pred),dfa.validation$appetency))
get_auc(tab_log)

#----------------------Extreme Gradient Boosting-------------------------------------------#
xgbGrid <- expand.grid(nrounds = c(100,200),  # this is n_estimators in the python code above
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree = seq(0.5, 0.9, length.out = 5),
                       ## The values below are default values in the sklearn-api. 
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1)

X_train = xgb.DMatrix(as.matrix(dfa.train[,-1]))
y_train = dfa.train$appetency
X_test = xgb.DMatrix(as.matrix(dfa.validation[,-1]))
y_test = dfa.validation$appetency

set.seed(0) 
xgb_model = train(
  X_train, y_train,  
  trControl = ctrl,
  tuneGrid = xgbGrid,
  method = "xgbTree")

#Prediction on validation dataset
predict_xgb = predict(xgb_model, X_test)
confusionMatrix(predict_xgb,y_test,positive="1")

#Prediction on test dataset
xgb_pred2 = predict(xgb_model, newdata = dfa.test,na.action = na.pass)
xgb_pred2

tab_xgb = as.matrix(table(predict_xgb,dfa.validation$appetency))
get_auc(tab_xgb)
