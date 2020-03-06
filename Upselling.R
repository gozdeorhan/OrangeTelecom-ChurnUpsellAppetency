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
#library(klaR)
library(klaR, lib.loc="/home/gorha001/my_RM_work")
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
#library(funModeling)
library(funModeling, lib.loc="/home/gorha001/my_RM_work")
library(rpart)
library(leaps)
library(MASS)
library(nnet)
#library(RANN)
library(RANN, lib.loc="/home/gorha001/my_RM_work")


#setwd('/Users/gozdeorhan/Desktop/ML_CW2')
setwd('/home/gorha001/my_RM_work')

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

dfu = cbind(train_x,train_y['upselling'])
dfu$upselling = as.factor(dfu$upselling)
dim(dfu)
summary(dfu)

#-------Create a new dataframe to be used to check informative missingness-------#

dfu.NA = dfu
for (i in colnames(dfu.NA)){                             #convert NA to 1/0
  if (sum(is.na(dfu.NA[,i]))>0) {
    dfu.NA[,paste(i,"NA",sep="")] = ifelse(is.na(dfu.NA[[i]]), "1", "0")
    dfu.NA = dfu.NA[,!names(dfu.NA) %in% i]
  }
}
dim(dfu.NA)

#----------------------------------------------------------------------------#

test.NA = test_x
for (i in colnames(test.NA)){                             #convert NA to 1/0
  if (sum(is.na(test.NA[,i]))>0) {
    test.NA[,paste(i,"NA",sep="")] = ifelse(is.na(test.NA[[i]]), "1", "0")
    test.NA = test.NA[,!names(test.NA) %in% i]
  }
}
dim(test.NA)

#----------------Decision tree to check informative missingness-----------------#

#args=list(split = "information") #split based on mse 
#dtree_u=rpart(upselling~.,data=dfu.NA,method="class",control=rpart.control(minsplit=30,cp=0.001),parms = args)
#varImp(dtree_u, surrogates = FALSE, competes = FALSE)

#It is observed that V36 and V4 have informative missingness

#add V36NA and V4NA with the main dataset
dfu$V36NA=dfu.NA$V36NA
dfu$V4NA=dfu.NA$V4NA

#add V36NA and V4NA with the test dataset
test_x$V36NA=test.NA$V36NA
test_x$V4NA=test.NA$V4NA


dfu[sapply(dfu, is.character)] = lapply(dfu[sapply(dfu, is.character)],as.factor) #convert char to factor
test_x[sapply(test_x, is.character)] = lapply(test_x[sapply(test_x, is.character)],as.factor) #convert char to factor

#------------------------------Split the dataset---------------------------------#

# Randomly shuffle the data
set.seed(234)
dfu<-dfu[sample(nrow(dfu)),]

#Splitting the data into training and testing
splitu <- createDataPartition(dfu$upselling, p = 0.70)[[1]]
dfu.train <- dfu[splitu,]
dfu.validation <- dfu[-splitu,]
dim(dfu.train)
dim(dfu.validation)

#-----------------------Check balance of dataset (upselling)---------------------#

#barplot(prop.table(table(dfu.train$upselling)),xlab = "Upselling distribution",ylab = "Frequency of the sample")

#-----------Remove correlated columns (train, validation, test)------------------#
# correlation calculation
# find numeric dataset
numeric_dat = dfu.train[sapply(dfu.train, is.numeric)]

#calculate correlation on numeric values
correlation_dat = cor(numeric_dat,use='pairwise.complete.obs')
corrplot.mixed(correlation_dat)

#find the subset of correlation matrix with specific threshold values
#column names that are above than certain correlation
corr_colnames = findCorrelation(correlation_dat,cutoff = 0.75,names = TRUE)

#remove those columns
dfu.train = dfu.train[, !(names(dfu.train) %in% corr_colnames)]
dfu.validation = dfu.validation[, !(names(dfu.validation) %in% corr_colnames)]
test_x = test_x[, !(names(test_x) %in% corr_colnames)]
dim(test_x)
dim(dfu.validation)
dim(dfu.train)

#----------------------Imputation (train)-----------------------#

#create dataframe with numerical column
num_u.dat = dfu.train[sapply(dfu.train, is.numeric)]
dfu.train.imp_n = preProcess(num_u.dat, method = c("knnImpute","center","scale")) ##KNN IMPUTATION for numeric values
dfu.train.num = predict(dfu.train.imp_n,num_u.dat)

#create dataframe with categorical column
categorical_u.dat = dfu.train[sapply(dfu.train, is.factor)]

#dfu.train.imp_c = knnImputation(categorical_u.dat) ##KNN IMPUTATION for categorical values
#write.csv(dfu.train.imp_c,file = "u_cate.knn.upselling.csv",row.names = FALSE) ##write in a file as it take too much time

dfu.train.imp_c = read.csv(file="u_cate.knn.upselling.csv", header=TRUE) #read the imputed file

dfu.train.imp_c = dfu.train.imp_c[c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30,31,29)]

dfu.train.imp_c$upselling = as.factor(dfu.train.imp_c$upselling)
dfu.train.imp_c$V36NA = as.factor(dfu.train.imp_c$V36NA)
dfu.train.imp_c$V4NA = as.factor(dfu.train.imp_c$V4NA)

dfu.train.imputed = cbind(dfu.train.num,dfu.train.imp_c)  ## combine both dataframe to create final train data
dim(dfu.train.imputed)

#----------------------Imputation (validation)-----------------------#
#create dataframe with numerical column
num_u.val.dat = dfu.validation[sapply(dfu.validation, is.numeric)]
dfu.val.imput.n = predict(dfu.train.imp_n,num_u.val.dat)

#create dataframe with categorical column
categorical_u.val.dat = dfu.validation[sapply(dfu.validation, is.factor)]

#dfu.val.imp_c = knnImputation(categorical_u.val.dat) ##KNN IMPUTATION for categorical values
#write.csv(dfu.val.imp_c,file = "u_valC.knn.upselling.csv",row.names = FALSE) ##write in a file as it take too much time

dfu.val.imp_c = read.csv(file="u_valC.knn.upselling.csv", header=TRUE) #read the imputed file

dfu.val.imp_c = dfu.val.imp_c[c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30,31,29)]

dfu.val.imp_c$upselling = as.factor(dfu.val.imp_c$upselling)
dfu.val.imp_c$V36NA = as.factor(dfu.val.imp_c$V36NA)
dfu.val.imp_c$V4NA = as.factor(dfu.val.imp_c$V4NA)

dfu.val.imputed = cbind(dfu.val.imput.n,dfu.val.imp_c)  ## combine both dataframe to create final train data
dim(dfu.val.imputed)

#----------------------Imputation (test)-----------------------#
#create dataframe with numerical column
num_u.test.dat = test_x[sapply(test_x, is.numeric)]
test.imput.n = predict(dfu.train.imp_n,num_u.test.dat)

#create dataframe with categorical column
categorical_u.test.dat = test_x[sapply(test_x, is.factor)]

#test.imp_c = knnImputation(categorical_u.test.dat) ##KNN IMPUTATION for categorical values
#write.csv(test.imp_c,file = "u_test.knn.upselling.csv",row.names = FALSE) ##write in a file as it take too much time

test.imp_c = read.csv(file="u_test.knn.upselling.csv", header=TRUE) #read the imputed file

test.imp_c$V36NA = as.factor(test.imp_c$V36NA)
test.imp_c$V4NA = as.factor(test.imp_c$V4NA)

test.imputed = cbind(test.imput.n,test.imp_c)  ## combine both dataframe to create final train data
dim(test.imputed)

#----------------Balance dataset (Train)----------------------------#

dfu.train.balance = SMOTE(upselling~., dfu.train.imputed, perc.over = 500, k = 5, perc.under = 200)       #balance trainset
as.data.frame(table(dfu.train.balance$upselling))                                     #check the balance of target column
as.data.frame(table(dfu.train$upselling))                                     #check the balance of target column
dfu.train.f = dfu.train.balance
dim(dfu.train.f)

#----------------Feature Selection(RELIEF)--------------------------#

#reliefVal_u = attrEval("upselling",data = dfu.train.f,
                     #estimator = "ReliefFequalK",
                     #ReliefIterations = 50)
#reliefVal_u.col = names(sort(abs(reliefVal_u), decreasing = TRUE))
#reliefVal_u.col

#-----------Feature Selection (RELIEF/Permutation)------------------#
#check importance of categorical value(balanced data) 

#reliefPerm_u = permuteRelief(x=dfu.train.f[,-ncol(dfu.train.f)],y=dfu.train.f[,ncol(dfu.train.f)],
                           #nperm = 500, estimator = "ReliefFequalK",
                           #ReliefIterations = 50)
#tail(reliefPerm$permutations)
#histogram(~value|Predictor, data = reliefPerm$permutations)
#perm.col_u=names(sort(abs(reliefPerm_u$standardized[which(abs(reliefPerm_u$standardized)>=1.6)]), decreasing=T))
#perm.col_u

#------------------Feature Selection(ROC)---------------------------#
#check importance of categorical value(balanced data) 

#rocVal_u = filterVarImp(x=dfu.train.f[,-ncol(dfu.train.f)],y=dfu.train.f[,ncol(dfu.train.f)])
#rocVal.order_u = order( rocVal_u$X.1, decreasing=TRUE )                 # Sort by the value of the variable importance:
#for (rn in 1:length(rocVal.order_u)) {
  #print(rownames(rocVal_u)[rocVal.order_u[rn]])
#}

#-------Feature Selection(information gain based on entropy)--------#

#info.score_u = information.gain(upselling~., dfu.train.f)
#subset_u=cutoff.k(info.score_u, 2)
#f_u=as.simple.formula(info.score_u$attr_importance, "upselling")
#print(f_u)
#infoScore.order_u = order( info.score_u$attr_importance, decreasing=TRUE )  # Sort by the value of the variable importance:
#for (rn in 1:length(infoScore.order_u)) {
  #print(rownames(info.score_u)[infoScore.order_u[rn]])
#}

#-----Select columns based on previous feature selection methods (Train)------#

keeps = c('upselling','V30','V112',"V84","V67","V4","V196","V187","V156","V126","V107","V96","V95","V93","V90","V36","V29","V224","V222","V212","V168","V165","V154","V150","V138","V118","V115","V114")
dfu.train_updated = dfu.train.f[keeps]
dim(dfu.train_updated)
colSums(is.na(dfu.train_updated))                                                 #check for missing value

#-----Select columns based on previous feature selection methods (Validation)------#

dfu.val.updated = dfu.val.imputed[keeps]
dim(dfu.val.updated)
colSums(is.na(dfu.val.updated))   

#-------Select columns based on previous feature selection methods (Test)---------#

testkeep2 = c('V30','V112',"V84","V67","V4","V196","V187","V156","V126","V107","V96","V95","V93","V90","V36","V29","V224","V222","V212","V168","V165","V154","V150","V138","V118","V115","V114")
test.imputed = test.imputed[testkeep2]
dim(test.imputed)
colSums(is.na(test.imputed)) 

#---chisquare--
#(categorical data)
categorical.dat = dfu.train_updated[sapply(dfu.train_updated, is.factor)] # find categorical dataset
numerical.dat = dfu.train_updated[sapply(dfu.train_updated, is.numeric)]

chi.res = chi.squared(upselling~.,categorical.dat)
sort.chi = order(chi.res$attr_importance,decreasing = TRUE)
for (rn in 1:length(sort.chi)) {
  print(rownames(chi.res)[sort.chi[rn]])
}


#--------------------Minimize labels in categorical columns (Train)-------------------#

categorical.dat = dfu.train_updated[sapply(dfu.train_updated, is.factor)]         #find categorical dataset
nummerical.dat = dfu.train_updated[sapply(dfu.train_updated, is.numeric)]         #find numerical dataset

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
    x= auto_grouping(data = categorical.dat, input = colnames(categorical.dat)[col], target = 'upselling',n_groups = 10,model = "kmeans",seed = 999)
    #print(x$recateg_results)
    newDF = merge(newDF, x$df_equivalence, by = colnames(categorical.dat)[col])
    testDF = merge(testDF, x$df_equivalence, by = colnames(categorical.dat)[col])
    head(testDF)
    newDF[columnName] = NULL
    index = index+1
  }
  testDF = testDF
}

dfu.train_updated = cbind(newDF,nummerical.dat)

dfu.train_updated$V4_rec = as.factor(dfu.train_updated$V4_rec)
dfu.train_updated$V187_rec = as.factor(dfu.train_updated$V187_rec)
dfu.train_updated$V156_rec = as.factor(dfu.train_updated$V156_rec)
dfu.train_updated$V126_rec = as.factor(dfu.train_updated$V126_rec)
dfu.train_updated$V93_rec = as.factor(dfu.train_updated$V93_rec)
dfu.train_updated$V90_rec = as.factor(dfu.train_updated$V90_rec)
dfu.train_updated$V224_rec = as.factor(dfu.train_updated$V224_rec)
dfu.train_updated$V165_rec = as.factor(dfu.train_updated$V165_rec)
dfu.train_updated$V138_rec = as.factor(dfu.train_updated$V138_rec)
dfu.train_updated$V118_rec = as.factor(dfu.train_updated$V118_rec)

dim(dfu.train_updated)
colSums(is.na(dfu.train_updated))  

#--------------------Minimize labels in categorical columns (Validation)-------------------#

categorical_val.dat = dfu.val.updated[sapply(dfu.val.updated, is.factor)]         #find categorical dataset
nummerical_val.dat = dfu.val.updated[sapply(dfu.val.updated, is.numeric)]         #find numerical dataset

newDF.Val = categorical_val.dat

for (col in 1:length(columnList)) {
  print(columnList[col])
  y = auto_grouping(data = categorical_val.dat, input = columnList[col], target = 'upselling',n_groups = 10,model = "kmeans",seed = 999)
  #print(y$recateg_results)
  newDF.Val = merge(newDF.Val, y$df_equivalence, by = columnList[col])
  newDF.Val[columnList[col]] = NULL
}
dfu.validation_updated = cbind(newDF.Val,nummerical_val.dat)

dfu.validation_updated$V4_rec = as.factor(dfu.validation_updated$V4_rec)
dfu.validation_updated$V187_rec = as.factor(dfu.validation_updated$V187_rec)
dfu.validation_updated$V156_rec = as.factor(dfu.validation_updated$V156_rec)
dfu.validation_updated$V126_rec = as.factor(dfu.validation_updated$V126_rec)
dfu.validation_updated$V93_rec = as.factor(dfu.validation_updated$V93_rec)
dfu.validation_updated$V90_rec = as.factor(dfu.validation_updated$V90_rec)
dfu.validation_updated$V224_rec = as.factor(dfu.validation_updated$V224_rec)
dfu.validation_updated$V165_rec = as.factor(dfu.validation_updated$V165_rec)
dfu.validation_updated$V138_rec = as.factor(dfu.validation_updated$V138_rec)
dfu.validation_updated$V118_rec = as.factor(dfu.validation_updated$V118_rec)

dim(dfu.validation_updated)
colSums(is.na(dfu.validation_updated))   

#--------------------Minimize labels in categorical columns (Test)-------------------#

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

newDF.test$V4_rec = as.factor(newDF.test$V4_rec)
newDF.test$V187_rec = as.factor(newDF.test$V187_rec)
newDF.test$V156_rec = as.factor(newDF.test$V156_rec)
newDF.test$V126_rec = as.factor(newDF.test$V126_rec)
newDF.test$V93_rec = as.factor(newDF.test$V93_rec)
newDF.test$V90_rec = as.factor(newDF.test$V90_rec)
newDF.test$V224_rec = as.factor(newDF.test$V224_rec)
newDF.test$V165_rec = as.factor(newDF.test$V165_rec)
newDF.test$V138_rec = as.factor(newDF.test$V138_rec)
newDF.test$V118_rec = as.factor(newDF.test$V118_rec)

dim(newDF.test)
colSums(is.na(newDF.test))
colSums(is.na(numerical.test))

#------------------------------Imputation (Test)---------------------------#

#newDF.test.imp_c = knnImputation(newDF.test) ##KNN IMPUTATION for categorical values
#write.csv(newDF.test.imp_c,file = "u_test.knn2.csv",row.names = FALSE) ##write in a file as it take too much time

test.imp_c = read.csv(file="u_test.knn2.csv", header=TRUE) #read the imputed file
test_x_updated = cbind(test.imp_c,numerical.test)
dim(test_x_updated)
colSums(is.na(test_x_updated)) 

#-----------------------Dummification (Train)-----------------------------#
#creating dummy variables for factor type variables

dummies = dummyVars( ~., data = dfu.train_updated[,-1],levelsOnly = FALSE)
data_dummy=predict(dummies,newdata= dfu.train_updated)
data_dummy=as.data.frame(data_dummy)
data_dummy=data.frame(dfu.train_updated$upselling,data_dummy)
names(data_dummy )[1]="upselling"
dfu.train = data_dummy 
dim(dfu.train)
sum(is.na(dfu.train))

#-------------------------Dummification (Validation)----------------------#
#creating dummy variables for factor type variables

dfu.validation_updated<-dfu.validation_updated[!(dfu.validation_updated$V196=="pybr"),]
dfu.validation_updated<-droplevels(dfu.validation_updated, exclude = if(anyNA(levels(dfu.validation_updated))) NULL else NA)

levels(dfu.validation_updated$V196) = levels(dfu.train_updated$V196)
levels(dfu.validation_updated$V29) = levels(dfu.train_updated$V29)
levels(dfu.validation_updated$V154) = levels(dfu.train_updated$V154)

data_dummyV=predict(dummies,newdata= dfu.validation_updated)
data_dummyV=as.data.frame(data_dummyV)
data_dummyV=data.frame(dfu.validation_updated$upselling,data_dummyV)
names(data_dummyV )[1]="upselling"
dfu.validation = data_dummyV 
dim(dfu.validation)
sum(is.na(dfu.validation))

#---------------------------Dummification (Test)-----------------------------#

levels(test_x_updated$V154) = levels(dfu.train_updated$V154)
levels(test_x_updated$V4_rec) = levels(dfu.train_updated$V4_rec)
levels(test_x_updated$V126_rec) = levels(dfu.train_updated$V126_rec)
levels(test_x_updated$V29) = levels(dfu.train_updated$V29)

data_dummyT=predict(dummies,newdata= test_x_updated)
data_dummyT=as.data.frame(data_dummyT)
dfu.test = data_dummyT 
dim(dfu.test)
sum(is.na(dfu.test))


#--------------------------FEATURE SELECTION (Part 2)-------------------------------------#
#--------------------------LASSO Regression----------------------------------#

#set.seed(123)
#x = model.matrix(upselling~., dfu.train)[,-1]
#y = ifelse(dfu.train$upselling == "1", 1, 0)
#cv.lasso = cv.glmnet(x, y, alpha = 1, family = "binomial")
#plot(cv.lasso)
#print(cv.lasso$lambda.min)
#print(coef(cv.lasso, cv.lasso$lambda.min))
#lasso.col = coef(cv.lasso, cv.lasso$lambda.min)
#write.table(as.data.frame.matrix(lasso.col), file = "u_lasso_c100.csv",row.names=FALSE,sep = ",",col.names = FALSE)

#update data frame after lasso 
c=2
keep_lassoCol = NULL
l.xlsx = read.delim("u_lasso_c100.csv",sep = ",",header = FALSE,stringsAsFactors = FALSE)
for (i in 1:length(l.xlsx[[1]])) {
  if (l.xlsx[[1]][i]!=0) {
    print(l.xlsx[[2]][i])
    keep_lassoCol[c] = l.xlsx[[2]][i]
    c=c+1
  }  
}

keep_lassoCol[1] = "upselling"
dfu.train = dfu.train[keep_lassoCol]
dim(dfu.train)

dfu.validation = dfu.validation[keep_lassoCol]
dim(dfu.validation)  

dfu.test = dfu.test[keep_lassoCol[-1]]
dim(dfu.test)

#---------------------Use VIF for multicolinearity--------------------------#

#set.seed(123)
#library(DAAG)
#logit = glm(upselling~.,data = dfu.train, family = binomial) # CREATE LOGISTIC MODEL WITH ALL VARIABLES
#vif.value = DAAG::vif(logit)
remove_col = c("V187_rec.group_3","V187_rec.group_9","V93_rec.group_10","V93_rec.group_5",'V93_rec.group_9','V90_rec.group_5','V224_rec.group_10','V224_rec.group_3','V224_rec.group_5','V224_rec.group_7','V118_rec.group_3','V118_rec.group_5')

dfu.train = dfu.train[, !(colnames(dfu.train) %in% remove_col)]
dim(dfu.train)

dfu.validation = dfu.validation[, !(colnames(dfu.validation) %in% remove_col)]
dim(dfu.validation)

dfu.test = dfu.test[, !(colnames(dfu.test) %in% remove_col)]
dim(dfu.test) 

#------------------------Feature Selection with P-value-----------------------------------------#

#set.seed(123)
#logit = glm(upselling~.,data = dfu.train, family = binomial) # CREATE LOGISTIC MODEL WITH ALL VARIABLES
#p.imp = data.frame(summary(logit)$coef[summary(logit)$coef[,4] <= .05, 4])
#write.table(p.imp, file = "u_logit.csv",row.names=TRUE,sep = ",",col.names = TRUE)

c=2
keep_pvalueCol = NULL
pValue.col = read.delim("u_logit.csv",sep = ",",header = FALSE,stringsAsFactors = FALSE)
for (i in 3:length(pValue.col[[1]])) {
  keep_pvalueCol[c] = pValue.col[[1]][i]
  c=c+1
}

keep_pvalueCol[1] = "upselling"
dfu.train = dfu.train[keep_pvalueCol]
dim(dfu.train)

dfu.validation = dfu.validation[keep_pvalueCol]
dim(dfu.validation)   #77 columns

dfu.test = dfu.test[keep_pvalueCol[-1]]
dim(dfu.test)

#----------------------------Feature Selection with RFE----------------------------------------------#

#control = rfeControl(functions=ldaFuncs, method="cv", number=3)
#results = rfe(dfu.train[,2:77], dfu.train[,1], sizes=c(2:77), rfeControl=control)
#summarize the results
#print(results)
#list the chosen features
#predictors(results)

#----------------------Feature Selection with Random Forest--------------------------------------#

#dfu.train.rf = randomForest(upselling ~ ., data=dfu.train, mtry=15,
                            #importance=TRUE, na.action=na.omit)
#varImpPlot(dfu.train.rf)
#rf.feature = dfu.train.rf$importance
#write.table(as.data.frame(rf.feature), file = "u_rf_f100.csv",row.names=TRUE,sep = ",",col.names = TRUE)

#-----------------------Columns after RFE and Random Forest (50)------------------------------------#

c=1
keep = NULL
columns.xlsx = read.delim("u_columns.csv",sep = ",",header = FALSE,stringsAsFactors = FALSE)
for (i in 1:length(columns.xlsx[[1]])) {
  keep[c] = columns.xlsx[[1]][i] #50 columns
    c=c+1
}  

dfu.train = dfu.train[keep]
dim(dfu.train) #50 columns

dfu.validation = dfu.validation[keep]
dim(dfu.validation)  #50 columns

dfu.test = dfu.test[keep[-1]]
dim(dfu.test)  #50 columns

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
dtree_fit = train(upselling ~., data = dfu.train, method = "rpart",
                  preProcess=c("center", "scale","BoxCox"),
                  trControl=ctrl,
                  tuneLength = 10)

#Prediction on validation dataset
dt_pred = predict(dtree_fit, newdata = dfu.validation)
confusionMatrix(dt_pred, dfu.validation$upselling,positive="1") 

#Prediction on test dataset
dt_pred2 = predict(dtree_fit, newdata = dfu.test,na.action = na.pass)
write.csv(dt_pred2,file = "upselling_prediction.csv",row.names = FALSE)

tab = as.matrix(table(dt_pred,dfu.validation$upselling))
get_auc(tab)

#-------------------Random Forest-----------------------#
set.seed(123)

#TRAIN MODEL
tunegrid=expand.grid(.mtry=7)
rf_fit = train(upselling~., data=dfu.train, 
               method='rf',  
               metric='Accuracy', 
               preProcess=c("center", "scale","BoxCox"),
               tuneGrid=tunegrid, 
               trControl=ctrl)

#Prediction on validation dataset
rf.predict=predict(rf_fit, newdata = dfu.validation)
confusionMatrix(rf.predict, dfu.validation$upselling,positive="1")

#Prediction on test dataset
rf_pred2 = predict(rf_fit, newdata = dfu.test,na.action = na.pass)
rf_pred2

tab_rf = as.matrix(table(rf.predict,dfu.validation$upselling))
get_auc(tab_rf)

#-------------------------Bagging----------------------------------#
set.seed(123)

tunegrid = expand.grid(.mtry=9)
bag_fit = train(upselling~., 
                data=dfu.train, 
                method='treebag', 
                metric='Accuracy',
                preProcess=c("center", "scale","BoxCox","spatialSign"),
                tunegrid = tunegrid,
                trControl=ctrl)

#Prediction on validation dataset
bag.predict = predict(bag_fit, newdata = dfu.validation )
confusionMatrix(bag.predict, dfu.validation$upselling,positive="1")

#Prediction on test dataset
bag_pred2 = predict(bag_fit, newdata = dfu.test,na.action = na.pass)
bag_pred2

tab_bag = as.matrix(table(bag.predict,dfu.validation$upselling))
get_auc(tab_bag)

#----------------------------SVM------------------------------------#
set.seed(123)
SVModel = train(upselling ~ ., data = dfu.train,
                method = "svmPoly",
                trControl= ctrl,
                tuneGrid = data.frame(degree = 1,
                                      scale = 1,
                                      C = 1),
                preProcess = c("scale","center"),
                na.action = na.omit)

#Prediction on validation dataset
vmPred<-predict(SVModel,dfu.validation[,names(dfu.validation)!="upselling"])
confusionMatrix(vmPred,dfu.validation$upselling,positive="1")

#Prediction on test dataset
svm_pred2 = predict(SVModel, newdata = dfu.test,na.action = na.pass)
svm_pred2

tab_svm = as.matrix(table(vmPred,dfu.validation$upselling))
get_auc(tab_svm)

#----------------------------KNN------------------------------------#
set.seed(123)
knn_fit = train(dfu.train[,-1], dfu.train$upselling, method = "knn",
                trControl=ctrl,
                preProcess = c("center", "scale","BoxCox","spatialSign"),
                tuneLength = 10)

#Prediction on validation dataset
knnPred = predict(knn_fit, newdata = dfu.validation)
confusionMatrix(knnPred,dfu.validation$upselling,positive="1")

#Prediction on test dataset
knn_pred2 = predict(knn_fit, newdata = dfu.test,na.action = na.pass)
knn_pred2

tab_knn = as.matrix(table(knnPred,dfu.validation$upselling))
get_auc(tab_knn)

#----------------------Naive Bayes Algorithm--------------------------#
set.seed(123)

nb_fit = train(dfu.train[,-1], dfu.train$upselling, method="nb", 
               preProcess=c("center", "scale","BoxCox","spatialSign"),
               trControl=ctrl)

#Prediction on validation dataset
nb_pred = predict(nb_fit, dfu.validation)
confusionMatrix(nb_pred, dfu.validation$upselling, positive="1")

#Prediction on test dataset
nb_pred2 = predict(nb_fit, newdata = dfu.test,na.action = na.pass)
nb_pred2

tab_nb = as.matrix(table(nb_pred,dfu.validation$upselling))
get_auc(tab_nb)

#----------------------Extreme Gradient Boosting-------------------------------------------#
xgbGrid <- expand.grid(nrounds = c(100,200),  # this is n_estimators in the python code above
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree = seq(0.5, 0.9, length.out = 5),
                       ## The values below are default values in the sklearn-api. 
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1)

X_train = xgb.DMatrix(as.matrix(dfu.train[,-1]))
y_train = dfu.train$upselling
X_test = xgb.DMatrix(as.matrix(dfu.validation[,-1]))
y_test = dfu.validation$upselling

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
xgb_pred2 = predict(xgb_model, newdata = dfu.test,na.action = na.pass)
xgb_pred2

tab_xgb = as.matrix(table(predict_xgb,dfu.validation$upselling))
get_auc(tab_xgb)

#----------------------Logistic Regression-------------------------------------------#
set.seed(123)
logit = glm(upselling~.,data=dfu.train,family = binomial)
glm.probs = predict(logit, dfu.validation, type='response')
glm.pred <- ifelse(glm.probs>0.5,'-1','1')
mean(glm.pred==dfu.validation$upselling)
threshold <- 0.5
confusionMatrix(factor(glm.pred),factor(dfu.validation$upselling),positive="1")

tab_log = as.matrix(table(factor(glm.pred),dfu.validation$upselling))
get_auc(tab_log)
