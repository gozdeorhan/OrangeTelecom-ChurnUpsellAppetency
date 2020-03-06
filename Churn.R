#----------------------------------------------------------------------------#

#Team Leader: Rubaida Easmin
#Nidya Esperanza Ballesteros Avila
#Gozde Orhan
#9/04/2019

#----------------------------------------------------------------------------#

#remove workspace
rm(list=ls())

#--------------------------Libraries----------------------------------------#
#setwd("/home/reasm001/RCode")
setwd("C:/Users/eruba/Documents/ML/data/")

#install.packages("corrplot")
#install.packages("mice")
#install.packages('DMwR')
#install.packages('FSelector')   
#install.packages("RWekajars")
#install.packages('rJava')
#install.packages("CORElearn")
#install.packages("AppliedPredictiveModeling")
#install.packages("funModeling")
#install.packages("Amelia")
#install.packages("naivebayes")

library(naivebayes)
library(caret)
library(corrplot)
library(reshape2)
library(dplyr)
library(DMwR)
library(FSelector)
library(rJava)
library(CORElearn)
library(AppliedPredictiveModeling)
library(randomForest)
library(funModeling)
library(rpart)
library(leaps)
library(MASS)


#----------------------------Read Datasets------------------------------------#
train_x=read.delim("train_X.csv", stringsAsFactors = FALSE, header=FALSE,na.strings=c(""," ",NA))
dim(train_x)

train_y=read.delim("train_Y.csv", stringsAsFactors = FALSE, header=TRUE,na.strings=c(""," ",NA))
dim(train_y)

test_x=read.delim("test_X.csv", stringsAsFactors = FALSE, header=FALSE,na.strings=c(""," ",NA))
dim(test_x)


#--------------------------Removing Missing Values >0.70 ------------------------#
miss_val_col = colSums(is.na(train_x))
sum(colMeans(is.na(train_x)) >=0.7)                 #number of columns with more than 70% missing values
train_x = train_x[,colMeans(is.na(train_x)) <= 0.7] #create new data frame after removing missing values (156 column)
dim(train_x)

#Remove columns from test set
test_x = test_x[, (names(test_x) %in% names(train_x))]# keep train set columns
dim(test_x)


#--------------------------Remove near zero and zero variance ------------------------#
predictorInfo = nearZeroVar(train_x,saveMetrics = TRUE)
rownames(predictorInfo)[predictorInfo$nzv]         #column names that have near zero variance (10 columns)
train_x = train_x[,!predictorInfo$nzv]             #remove nzv column from dataset
dim(train_x)    #33001    64

#Remove columns from test set
test_x = test_x[, (names(test_x) %in% names(train_x))]#keep train set columns
dim(test_x)   #16999    64


#--------------------------Combine train set with target ------------------------#
dfc = cbind(train_x,train_y["churn"])
dfc$churn = as.factor(dfc$churn)                  #convert target column int to factor
dim(dfc)
summary(dfc)

#-------Create a new dataframe to be used to check informative missingness-------#
dfc.NA = dfc
for (i in colnames(dfc.NA)){                             #convert NA to 1/0
  if (sum(is.na(dfc.NA[,i]))>0) {
    dfc.NA[,paste(i,"NA",sep="")] = ifelse(is.na(dfc.NA[[i]]), "1", "0")
    dfc.NA = dfc.NA[,!names(dfc.NA) %in% i]
  }
}
dim(dfc.NA)


#------------------Decision tree to check informative missingness-----------------#
args=list(split = "information") #split based on mse 
dtree=rpart(churn~.,data=dfc.NA,method="class",control=rpart.control(minsplit=30,cp=0.001),parms = args)
varImp(dtree, surrogates = FALSE, competes = FALSE)

#It is observed that V36 and V84 have informative missingness

#add V36 and V84 with the main dataset
dfc$V36NA=dfc.NA$V36NA
dfc$V84NA=dfc.NA$V84NA


#--------------------------------Split the dataset---------------------------------#
set.seed(234)
dfc = dfc[sample(nrow(dfc)),]                                 # Randomly shuffle the data
split = createDataPartition(y=dfc$churn, p = 0.70,list = FALSE)
dfc.train = dfc[split,]
dfc.validation = dfc[-split,]

dfc.train[sapply(dfc.train, is.character)] = lapply(dfc.train[sapply(dfc.train, is.character)],as.factor) #convert char to factor
dfc.validation[sapply(dfc.validation, is.character)] = lapply(dfc.validation[sapply(dfc.validation, is.character)],as.factor)
test_x[sapply(test_x, is.character)] = lapply(test_x[sapply(test_x, is.character)],as.factor) 

#summary(dfc.train)
#summary(dfc.validation)

#-------------------------------Check balance of dataset-----------------------#
barplot(prop.table(table(dfc.train$churn)),xlab = "Distribution of target variable (Churn)",ylab = "Frequency of the sample")



#--------------------------------correlation calculation-----------------------#
numeric.dat = dfc.train[sapply(dfc.train, is.numeric)]         # find numeric dataset
correlation_dat = cor(numeric.dat,use='pairwise.complete.obs') #calculate correlation on numeric values
corrplot.mixed(correlation_dat)                                # plot correlation matrix
corr_colnames = findCorrelation(correlation_dat,cutoff = 0.75,names = TRUE) #column names above threshold (8 columns)
dfc.train = dfc.train[, !(names(dfc.train) %in% corr_colnames)]#remove those columns
dim(dfc.train)

#validation set
dfc.validation = dfc.validation[, !(names(dfc.validation) %in% corr_colnames)]#remove those columns
dim(dfc.validation)

#test set
test_x = test_x[,(names(test_x) %in% names(dfc.train))]#keep train set columns
dim(test_x)


#--------------------------------Imputation(Train)---------------------------------#
#create dataframe with numerical column
num.dat = dfc.train[sapply(dfc.train, is.numeric)]
numeric.imputation = preProcess(num.dat, method = c("knnImpute","center","scale")) ##KNN IMPUTATION for numeric values
dfc.train.num = predict(numeric.imputation,num.dat)

#create dataframe with categorical column
categorical.dat = dfc.train[sapply(dfc.train, is.factor)]
#dfc.train.imp_c = knnImputation(categorical.dat,k =6) ##KNN IMPUTATION for categorical values
#write.csv(dfc.train.imp_c,file = "cate.knn6.csv",row.names = FALSE) ##write in a file as it take too much time
dfc.train.imp_c = read.csv(file="cate.knn6.csv", header=TRUE) #read the imputed file
dfc.train.imp_c =dfc.train.imp_c[c(31,1:30)]
dfc.train.imp_c$churn = as.factor(dfc.train.imp_c$churn)
dfc.train.imp_c$V84NA = as.factor(dfc.train.imp_c$V84NA)
dfc.train.imp_c$V36NA = as.factor(dfc.train.imp_c$V36NA)
dfc.train.imputed = cbind(dfc.train.imp_c,dfc.train.num)  ## combine both dataframe to create final train data
dim(dfc.train.imputed)

#--------------------------------Imputation(Validation)---------------------------------#
#create dataframe with numerical column
num.val.dat = dfc.validation[sapply(dfc.validation, is.numeric)]
dfc.val.imput.n = predict(numeric.imputation,num.val.dat)

#create dataframe with categorical column
categorical.val.dat = dfc.validation[sapply(dfc.validation, is.factor)]
#dfc.val.imp_c = knnImputation(categorical.val.dat,k = 6) ##KNN IMPUTATION for categorical values
#write.csv(dfc.val.imp_c,file = "valC.knn6.csv",row.names = FALSE) ##write in a file as it take too much time
dfc.val.imp_c = read.csv(file="valC.knn6.csv", header=TRUE) #read the imputed file
dfc.val.imp_c = data.frame(dfc.val.imp_c$churn,dfc.val.imp_c[,-29])
names(dfc.val.imp_c)[1] = "churn"
dfc.val.imp_c$churn = as.factor(dfc.val.imp_c$churn)
dfc.val.imp_c$V84NA = as.factor(dfc.val.imp_c$V84NA)
dfc.val.imp_c$V36NA = as.factor(dfc.val.imp_c$V36NA)
dfc.val.imputed = cbind(dfc.val.imp_c,dfc.val.imput.n)  ## combine both dataframe to create final train data
dim(dfc.val.imputed)


#--------------------------------Imputation(Test)---------------------------------#
num.test = test_x[sapply(test_x, is.numeric)]           ## imputation on numerical value
test_x.imput.n = predict(numeric.imputation,num.test)

categorical.test1 = test_x[sapply(test_x, is.factor)]
test.imputed = cbind(categorical.test1,test_x.imput.n)  ## combine both categorical and numerical dataframe 
dim(test_x)


#--------------------------------Balance dataset (TRAIN)----------------------------#
as.data.frame(table(dfc.train.imputed$churn))                                     #check the balance of target column
dfc.train.balance = SMOTE(churn~., dfc.train.imputed, perc.over = 500, k = 5, perc.under = 200)       #balance trainset
as.data.frame(table(dfc.train.balance$churn))                                     #check the balance of target column
dfc.train.f = dfc.train.balance


#--------------------------------------Feature Selection(RELIEF)--------------------------#
reliefVal = attrEval("churn",data = dfc.train.f,
                     estimator = "ReliefFequalK",
                     ReliefIterations = 50)
reliefVal.col = names(sort(abs(reliefVal), decreasing = TRUE)[1:40])

#-----------------------Feature Selection (RELIEF/Permutation)-----------------------------#
reliefPerm = permuteRelief(x = dfc.train.f[,-1],y = dfc.train.f$churn,
                           nperm = 100, estimator = "ReliefFequalK",
                           ReliefIterations = 50)
perm.col=names(sort(abs(reliefPerm$standardized[which(abs(reliefPerm$standardized)>=1.6)]), decreasing=T))
#tail(reliefPerm$permutations)
#histogram(~value|Predictor, data = reliefPerm$permutations)

#----------------------------------Feature Selection(ROC)---------------------------------#
rocVal = filterVarImp( x=dfc.train.f[,-1], y=dfc.train.f$churn )
rocVal.order = order( rocVal$X.1, decreasing=TRUE )                 # Sort by the value of the variable importance:
for (rn in 1:length(rocVal.order)) {
  print(rownames(rocVal)[rocVal.order[rn]])
}
#----------------------Feature Selection(information gain based on entropy)-------------------#
info.score = information.gain(churn~., dfc.train.f)
subset=cutoff.k(info.score, 2)
f=as.simple.formula(info.score$attr_importance, "churn")
infoScore.order = order( info.score$attr_importance, decreasing=TRUE )  # Sort by the value of the variable importance:
for (rn in 1:length(infoScore.order)) {
  print(rownames(info.score)[infoScore.order[rn]])
}


#--------------------Select columns based on attrEval, permRelief, filterVarImp (Train)-------------#
#remove 28 column
keeps = c("churn","V187","V84","V67","V42","V116","V118","V156","V107","V36","V115","V21","V192","V93","V4","V90",
          "V138","V224","V165","V119","V196","V125","V84NA","V121","V190","V29","V37","V95","V134","V15","V71","V126")
dfc.train_updated = dfc.train.f[keeps]
dim(dfc.train_updated)
colSums(is.na(dfc.train_updated))                                                 #check for missing value

#--------------------Select columns based on attrEval, permRelief, filterVarImp (Validation)-------------#
dfc.val.imputed = dfc.val.imputed[keeps]
dim(dfc.val.imputed)
colSums(is.na(dfc.val.imputed))        

#--------------------Select columns based on attrEval, permRelief, filterVarImp (Test)-------------#
test.imputed = test.imputed[,(names(test.imputed) %in% names(dfc.train_updated))]#keep train set columns
dim(test.imputed)
colSums(is.na(test.imputed))     

#------------------select categorical and numerical set (Train)------------------------#
categorical.dat = dfc.train_updated[sapply(dfc.train_updated, is.factor)]         # find categorical dataset
nummerical.dat = dfc.train_updated[sapply(dfc.train_updated, is.numeric)]



#-----------------------------chi square test (categorical data)-----------------------#
chi.res = chi.squared(churn~.,categorical.dat)
sort.chi = order(chi.res$attr_importance,decreasing = TRUE)
for (rn in 1:length(sort.chi)) {
  print(rownames(chi.res)[sort.chi[rn]])
}

#-------------------------minimize levels in categorical columns(Train)-------------------#
newDF=categorical.dat
testDF = categorical.dat
columnList = NULL
index = 1
for (col in 2:length(colnames(categorical.dat))) {
  col.levels = nlevels(categorical.dat[[colnames(categorical.dat)[col]]])
  #print(col.levels)
  if(col.levels>1000){
    print(colnames(categorical.dat)[col])
    columnName = colnames(categorical.dat)[col]
    columnList [[index]] = columnName
    x= auto_grouping(data = categorical.dat, input = colnames(categorical.dat)[col], target = 'churn',n_groups = 5,model = "kmeans",seed = 999)
    #print(x$recateg_results)
    newDF = merge(newDF, x$df_equivalence, by = colnames(categorical.dat)[col])
    testDF = merge(testDF, x$df_equivalence, by = colnames(categorical.dat)[col])
    head(testDF)
    newDF[columnName] = NULL
    index = index+1
  }
  testDF = testDF
}

dfc.train_updated = cbind(newDF,nummerical.dat)
dim(dfc.train_updated)



#--------------------minimize levels in categorical columns(Validation)-------------------#
categorical_val.dat = dfc.val.imputed[sapply(dfc.val.imputed, is.factor)]         # find categorical dataset
nummerical_val.dat = dfc.val.imputed[sapply(dfc.val.imputed, is.numeric)]
newDF.Val = categorical_val.dat

for (col in 1:length(columnList)) {
  print(columnList[col])
  #columnName = colnames(categorical.dat)[col]
  y = auto_grouping(data = categorical_val.dat, input = columnList[col], target = 'churn',n_groups = 5,model = "kmeans",seed = 999)
  print(y$recateg_results)
  newDF.Val = merge(newDF.Val, y$df_equivalence, by = columnList[col])
  newDF.Val[columnList[col]] = NULL
}
dfc.validation_updated = cbind(newDF.Val,nummerical_val.dat)
dim(dfc.validation_updated)
colSums(is.na(dfc.validation_updated))                               


#--------------------minimize levels in categorical columns(Test)-------------------#

cat.test = test.imputed[sapply(test.imputed, is.factor)]         # find categorical dataset
numerical.test = test.imputed[sapply(test.imputed, is.numeric)]
newDF.test = cat.test

for (i in 1:length(columnList)) {   #length(columnList)
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
colSums(is.na(newDF.test))  


#------------------------------Imputation(test)---------------------------#
#create dataframe with categorical column
#categorical.test = test_x_updated[sapply(test_x_updated, is.factor)]
newDF.test[sapply(newDF.test, is.character)] = lapply(newDF.test[sapply(newDF.test, is.character)],as.factor) 
#test_x.imp_c = knnImputation(newDF.test,k = 6) ##KNN IMPUTATION for categorical values
#write.csv(test_x.imp_c,file = "test.knn6.csv",row.names = FALSE) ##write in a file as it take too much time

test.imp_c = read.csv(file="test.knn6.csv", header=TRUE) #read the imputed file
test_x_updated = cbind(test.imp_c,numerical.test)       #Final test set
dim(test_x_updated)
colSums(is.na(test_x_updated))   



#------------------------------Feature Selectionon Train set (Gradient Boosting)-----------------------------#
library("gbm")
y = ifelse(dfc.train_updated$churn == "1", 1, 0)
gbm.fit = gbm(y~.,n.trees=1000,distribution="bernoulli", verbose=TRUE,interaction.depth = 3,data=dfc.train_updated,shrinkage = 0.01)
summary.gbm(gbm.fit)

keep.gbm = c("churn","V138","V156","V42","V192","V36","V67","V21","V84","V15","V71","V37","V134","V126_rec","V224_rec","V121","V4_rec","V95","V107","V119","V29","V190","V118_rec","V125","V196","V187_rec")
dfc.train_updated = dfc.train_updated[keep.gbm]
dim(dfc.train_updated)  #26 columns
colSums(is.na(dfc.train_updated))                                                 #check for missing value


#---------------------------Feature Selectionon Validation set (Gradient Boosting)---------------------------#
dfc.validation_updated = dfc.validation_updated[keep.gbm]
dim(dfc.validation_updated)
colSums(is.na(dfc.validation_updated))  

#---------------------------Feature Selectionon Test set (Gradient Boosting)---------------------------#
test_x_updated = test_x_updated[keep.gbm[-1]]
dim(test_x_updated)
colSums(is.na(test_x_updated))   


#----------------------------------------Dummification(Train)------------------------------------#
#creating dummy variables for factor type variables
dummies = dummyVars( ~., data = dfc.train_updated[,-1],levelsOnly = FALSE)
data_dummy=predict(dummies,newdata= dfc.train_updated)
data_dummy=as.data.frame(data_dummy)
data_dummy=data.frame(dfc.train_updated$churn,data_dummy)
names(data_dummy )[1]="churn"
dfc.train = data_dummy 
dim(dfc.train)
sum(is.na(dfc.train))

#--------------------------------------Dummification(Validation)-----------------------------------#
#creating dummy variables for factor type variables
levels(dfc.validation_updated$V138) = levels(dfc.train_updated$V138)
levels(dfc.validation_updated$V156) = levels(dfc.train_updated$V156)
levels(dfc.validation_updated$V190) = levels(dfc.train_updated$V190)

data_dummyV=predict(dummies,newdata= dfc.validation_updated)
data_dummyV=as.data.frame(data_dummyV)
data_dummyV=data.frame(dfc.validation_updated$churn,data_dummyV)
names(data_dummyV )[1]="churn"
dfc.validation = data_dummyV 
dim(dfc.validation)
sum(is.na(dfc.validation))

#--------------------------------------Dummification(Test)-----------------------------------------#
levels(test_x_updated$V138) = levels(dfc.train_updated$V138)
levels(test_x_updated$V156) = levels(dfc.train_updated$V156)
levels(test_x_updated$V29) = levels(dfc.train_updated$V29)
levels(test_x_updated$V190) = levels(dfc.train_updated$V190)


data_dummyT=predict(dummies,newdata= test_x_updated,na.action = na.pass)
data_dummyT=as.data.frame(data_dummyT)
dfc.test = data_dummyT 
dim(dfc.test)
sum(is.na(dfc.test))

#----------------------------Feature Selection (LASSO Regression)------------------------------------#
library(glmnet)
set.seed(123)
x = model.matrix(churn~., dfc.train)[,-1]
y = ifelse(dfc.train$churn == "1", 1, 0)
cv.lasso = cv.glmnet(x, y, alpha = 1, family = "binomial")
plot(cv.lasso)
print(cv.lasso$lambda.min)
print(coef(cv.lasso, cv.lasso$lambda.min))
lasso.col = coef(cv.lasso, cv.lasso$lambda.min)
#write.table(as.data.frame.matrix(lasso.col), file = "lasso_c100.csv",row.names=FALSE,sep = ",",col.names = FALSE)

#update data frame after lasso (remove 193 columns)
c=2
keep_lassoCol = NULL
l.xlsx = read.delim("lasso_c100.csv",sep = ",",header = FALSE,stringsAsFactors = FALSE)
for (i in 1:length(l.xlsx[[2]])) {
  if (l.xlsx[[2]][i]!=0) {
    print(l.xlsx[[1]][i])
    keep_lassoCol[c] = l.xlsx[[1]][i]
    print(c)
    c=c+1
  }  
}

keep_lassoCol[1] = "churn"
dfc.train = dfc.train[keep_lassoCol]
dim(dfc.train)

#update validation set
dfc.validation = dfc.validation[keep_lassoCol]
dim(dfc.validation)   #646 col 

#update test set
dfc.test = dfc.test[keep_lassoCol[-1]]
dim(dfc.test)   #646 col

#--------------------------------Use VIF for multicolinearity------------------------------------#
set.seed(123)
logit = glm(churn~.,data = dfc.train, family = binomial) # CREATE LOGISTIC MODEL WITH ALL VARIABLES
vif.value = DAAG::vif(logit)

remove_col = c("V196.9_Y1","V196.HLqf","V190.62ZfA6x0SiWOX","V29.r_7E")
dfc.train = dfc.train[, !(colnames(dfc.train) %in% remove_col)]
dim(dfc.train) #642 

#update validation set
dfc.validation = dfc.validation[, !(colnames(dfc.validation) %in% remove_col)]
dim(dfc.validation)

#update test set
dfc.test = dfc.test[, !(colnames(dfc.test) %in% remove_col)]
dim(dfc.test) 


#----------------------------Feature Selection (C5.0)------------------------------------#
library(C50)
model_c5 = C5.0(churn ~ ., data=dfc.train,trials=50,
                control = C5.0Control(                        
                  noGlobalPruning = T,
                  CF=0.8,
                  minCases=10,
                  sample = 0.80,
                  winnow=F,
                  earlyStopping=T
                ))
C5imp(model_c5)

c=2
keep_c5Col = NULL
c5.col = read.delim("C5.csv",sep = ",",header = FALSE,stringsAsFactors = FALSE)
for (i in 1:length(c5.col[[2]])) {
  if (c5.col[[2]][i]!=0) {
    print(c5.col[[1]][i])
    keep_c5Col[c] = c5.col[[1]][i]
    print(c)
    c=c+1
  }  
}
#update train set
keep_c5Col[1] = "churn"
dfc.train = dfc.train[keep_c5Col]
dim(dfc.train)

#update validation set
dfc.validation = dfc.validation[keep_c5Col]
dim(dfc.validation)   #438 columns

#update test set
dfc.test = dfc.test[keep_c5Col[-1]]
dim(dfc.test)

#------------------------------Feature Selection (P-Value)------------------------------------#
set.seed(123)
logit = glm(churn~.,data = dfc.train, family = binomial) # CREATE LOGISTIC MODEL WITH ALL VARIABLES
p.imp = data.frame(summary(logit)$coef[summary(logit)$coef[,4] <= .05, 4])
write.table(p.imp, file = "logit.csv",row.names=TRUE,sep = ",",col.names = TRUE)

c=2
keep_pvalueCol = NULL
pValue.col = read.delim("logit.csv",sep = ",",header = FALSE,stringsAsFactors = FALSE)
for (i in 1:length(pValue.col[[1]])) {
  keep_pvalueCol[c] = pValue.col[[1]][i]
  c=c+1
}

#update train set
keep_pvalueCol[1] = "churn"
dfc.train = dfc.train[keep_pvalueCol]
dim(dfc.train)

#update validation set
dfc.validation = dfc.validation[keep_pvalueCol]
dim(dfc.validation)   #166 columns

#update test set
dfc.test = dfc.test[keep_pvalueCol[-1]]
dim(dfc.test)

#------------------------------Feature Selection (Random Forest)------------------------------------#
dfc.train.rf = randomForest(churn ~ ., data=dfc.train, mtry=15,
                            importance=TRUE, na.action=na.omit)
varImpPlot(dfc.train.rf)
rf.feature = dfc.train.rf$importance
write.table(as.data.frame(rf.feature), file = "rf_f100.csv",row.names=TRUE,sep = ",",col.names = TRUE)

#--------------------------Feature Selection (Recursive Feature Elimination)--------------------------#
control = rfeControl(functions=ldaFuncs, method="cv", number=3)
results = rfe(dfc.train[,2:100], dfc.train[,1], sizes=c(2:100), rfeControl=control)
#summarize the results
print(results)
#list the chosen features
predictors(results)

#-------------------------Select Columns from top 50 (After RF and RFE)-----------------------#
keep.rffe = c("churn","V118_recgroup_3","V125.kIsH","V224_recgroup_3","V156.0Xwj","V126_recgroup_3","V4_recgroup_1","V224_recgroup_4","V126_recgroup_5","V118_recgroup_2","V4_recgroup_5","V4_recgroup_3","V126_recgroup_1",
              "V224_recgroup_2","V190.Ie_5MZs","V187_recgroup_3","V29.Zy3gnGM","V118_recgroup_5","V126_recgroup_4","V118_recgroup_4","V4_recgroup_4","V224_recgroup_5",
              "V187_recgroup_1","V187_recgroup_5","V29.xwM2aC7IdeMC0","V37","V190.JBfYVit4g8","V29.F2FcTt7IdMT_v","V119.RVjC","V190.XfqtO3UdzaXh_","V138.dTGmfo8zhV",
              "V190.UbxQ8lZ","V119.AKLO","V138.8I1q9ayE15","V138.IXSgUHShse","V138.MtpBcmzkzH","V119.jakt","V119.4nnx","V119.MBhA","V190.DmYOBF5GfjJxb","V190.lCToAAt",
              "V138.1JGqrQKzJV","V138.CEat0G8rTN","V138.J9Vr4RQZiT","V156.WkTj","V190._5OXC8MSLt","V119.RcM7","V119.TjV7")

#update train set
dfc.train = dfc.train[keep.rffe]
dim(dfc.train)         #48 columns

#update validation set
dfc.validation = dfc.validation[keep.rffe]
dim(dfc.validation)    #48 columns

#update test set
dfc.test = dfc.test[keep.rffe[-1]]
dim(dfc.test)         #48 columns


#-----------------------------------------Model Analysis-------------------------------------------#

# Function for measuring AUC
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


#-----------------------------------Best three models-----------------------------------#
#--------------------------------------Decision Tree------------------------------------#

trctrl = trainControl(method = "cv", number = 15)
set.seed(3333)
dtree_fit = train(churn ~., data = dfc.train, method = "rpart",
                  parms = list(split = "information"),
                  preProcess=c("center", "scale","BoxCox"),
                  trControl=trctrl,
                  tuneLength = 10)
dtree_fit
test_pred = predict(dtree_fit, newdata = dfc.validation)
confusionMatrix(test_pred, dfc.validation$churn)  

#AUC Score
tab = as.matrix(table(test_pred,dfc.validation$churn))
get_auc(tab)
roccurve = roc(as.numeric(test_pred), as.numeric(dfc.validation$churn))
plot(roccurve)

#prediction on test dataset
test_pred2 = predict(dtree_fit, newdata = dfc.test,na.action = na.pass)
test_pred2
write.table(test_pred2, file = "test_Y.csv", row.names=FALSE, sep="\t",col.names = c("churn"))

#----------------------------K-Nearest Neighbour---------------------------------------#
set.seed(234)
#install.packages("class")
library("class")
knn_fit = knn(train = dfc.train, test = dfc.validation, cl = dfc.train$churn, k = 3)
confusionMatrix(knn_fit ,dfc.validation$churn)

#AUC Score
tab = table(knn_fit ,dfc.validation$churn)
get_auc(tab)

#----------------------Naive Based Algorithm--------------------------#
library(klaR)
ctrl = trainControl(method="cv", 15)
set.seed(123)
nb_fit = train(dfc.train[,-1], dfc.train$churn, method="nb", 
               preProcess=c("center", "scale","BoxCox","spatialSign"),
               trControl=ctrl)
nb_pred = predict(nb_fit, dfc.validation)
confusionMatrix(nb_pred, dfc.validation$churn, positive="1")

#AUC Score
tab = as.matrix(table(nb_pred,dfc.validation$churn))
get_auc(tab)

#--------------------------------------Other Models-------------------------------------#

#--------------------------------------Logistic Regression------------------------------#
logit = glm(churn~.,data = dfc.train, family = binomial)
glm.probs =predict (logit ,dfc.validation, type="response")
glm.pred <- ifelse(glm.probs > 0.5, "-1", "1")
mean(glm.pred==dfc.validation$churn)
threshold <- 0.5
confusionMatrix(factor(glm.pred),factor(dfc.validation$churn))
tab = as.matrix(table(glm.pred,dfc.validation$churn))
get_auc(tab)

#---------------------------------------Random Forest--------------------------------#
set.seed(123)
fitControl=trainControl(
  method = "cv",
  number = 10)
tunegrid=expand.grid(.mtry=7)
rf_fit = train(churn~., data=dfc.train, 
               method='rf',  
               metric='Accuracy', 
               preProcess=c("center", "scale","BoxCox"),
               tuneGrid=tunegrid, 
               trControl=fitControl)

rf.predict=predict(rf_fit, newdata = dfc.validation)
confusionMatrix(rf.predict, dfc.validation$churn)

#AUC Score
tab = as.matrix(table(rf.predict,dfc.validation$churn))
get_auc(tab)


#-------------------------------------------Bagging-------------------------------------------#
set.seed(123)
ctrl = trainControl(method = "cv", number = 15)
tunegrid = expand.grid(.mtry=47)
bag_fit = train(churn~., 
                data=dfc.train, 
                method='treebag', 
                metric='Accuracy',
                preProcess=c("center", "scale","BoxCox"),
                tunegrid = tunegrid,
                trControl=ctrl)

bag.predict = predict(bag_fit, newdata = dfc.validation )
confusionMatrix(bag.predict, dfc.validation$churn)

#AUC Score
tab = as.matrix(table(bag.predict,dfc.validation$churn))
get_auc(tab)


#----------------------Extreme Gradient Descent----------------------------------#
library(xgboost)
ctrl = trainControl(method = "cv", number = 10)
xgbGrid <- expand.grid(nrounds = c(100,200),  # this is n_estimators in the python code above
                       max_depth = c(10, 15, 20),
                       colsample_bytree = seq(0.5, 0.9, length.out = 5),
                       ## The values below are default values in the sklearn-api. 
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1
)

X_train = xgb.DMatrix(as.matrix(dfc.train[,-1]))
y_train = dfc.train$churn
X_test = xgb.DMatrix(as.matrix(dfc.validation[,-1]))
y_test = dfc.validation$churn

set.seed(0) 
xgb_model = train(
  X_train, y_train,  
  trControl = ctrl,
  tuneGrid = xgbGrid,
  method = "xgbTree"
)

predicted = predict(xgb_model, X_test)
confusionMatrix(predicted,y_test)

#AUC Score
tab = as.matrix(table(predicted,dfc.validation$churn))
get_auc(tab)

#-----------------------------------Support Vector Machine--------------------------------#
set.seed(1)
SVModel = train(churn ~ ., data = dfc.train,
                method = "svmPoly",
                trControl= ctrl,
                tuneGrid = data.frame(degree = 1,
                                      scale = 1,
                                      C = 1),
                preProcess = c("scale","center"),
                na.action = na.omit
)
vmPred<-predict(SVModel,dfc.validation[,names(dfc.validation)!="churn"])
confusionMatrix(vmPred,dfc.validation$churn)

#AUC Score
tab = as.matrix(table(SVModel,dfc.validation[,names(dfc.validation)!="churn"]))
get_auc(tab)