## --------------------------------------------------------------------------------------------------------
banks <- readr::read_csv("bank_personal_loan.csv")


## --------------------------------------------------------------------------------------------------------
#View(banks)
library(corrplot)

correlation_matrix <- cor(banks)

# Print correlation matrix
corrplot(correlation_matrix, method = "color", 
         type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)


## --------------------------------------------------------------------------------------------------------

library("GGally")
library("dplyr")
banks$Personal.Loan <- as.factor(banks$Personal.Loan)
ggpairs(banks |> 
          select( Personal.Loan, Family,CCAvg,Mortgage,Income, Education,CD.Account),
        aes(color = Personal.Loan)
        )


## --------------------------------------------------------------------------------------------------------
library("skimr")
skimr::skim(banks)


## --------------------------------------------------------------------------------------------------------
banks$Personal.Loan <- as.factor(banks$Personal.Loan)
banks$CreditCard <- as.factor(banks$CreditCard)
banks$Online <- as.factor(banks$Online)
banks$CD.Account <- as.factor(banks$CD.Account)
banks$Securities.Account <- as.factor(banks$Securities.Account)
#banks$ZIP.Code <- as.factor(banks$ZIP.Code)
#banks$Family <- as.factor(banks$Family)
#banks$Education <- as.factor(banks$Education)


## --------------------------------------------------------------------------------------------------------



## --------------------------------------------------------------------------------------------------------
DataExplorer::plot_bar(banks,ncol=3)


## --------------------------------------------------------------------------------------------------------
DataExplorer::plot_histogram(banks,ncol=3)


## --------------------------------------------------------------------------------------------------------
DataExplorer::plot_boxplot(banks, by="Personal.Loan",ncol=3)


## --------------------------------------------------------------------------------------------------------



## --------------------------------------------------------------------------------------------------------
library("data.table")
library("mlr3verse")


## --------------------------------------------------------------------------------------------------------
set.seed(212) # set seed for reproducibility
loan_task <- TaskClassif$new(id = "Bank_Loan",
                               backend = banks, # <- NB: no na.omit() this time
                               target = "Personal.Loan",
                               positive = "1")


## --------------------------------------------------------------------------------------------------------
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(loan_task)


## --------------------------------------------------------------------------------------------------------
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart <- lrn("classif.rpart", predict_type = "prob")
lrn_svm   <- lrn("classif.svm", predict_type = "prob")
pl_svm <- po("encode") %>>%
  po(lrn_svm)
lrn_qda  <- lrn("classif.qda", predict_type = "prob")


## --------------------------------------------------------------------------------------------------------
res <- benchmark(data.table(
  task       = list(loan_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    pl_svm,
                    lrn_qda
                    #lrn_ranger,
                    #lrn_xgboost
                    ),
  resampling = list(cv5)
), store_models = TRUE)                       


## --------------------------------------------------------------------------------------------------------
res
res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))


## --------------------------------------------------------------------------------------------------------
# eg get the trees (2nd model fitted), by asking for second set of resample
# results
trees <- res$resample_result(2)

# Then, let's look at the tree from first CV iteration, for example:
tree1 <- trees$learners[[1]]

# This is a fitted rpart object, so we can look at the model within
tree1_rpart <- tree1$model

# If you look in the rpart package documentation, it tells us how to plot the
# tree that was fitted
plot(tree1_rpart, compress = TRUE, margin = 0.1)
text(tree1_rpart, use.n = TRUE, cex = 0.7)


## --------------------------------------------------------------------------------------------------------
i=3
plot(res$resample_result(2)$learners[[i]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[i]]$model, use.n = TRUE, cex = 0.8)


## --------------------------------------------------------------------------------------------------------
# Enable cross validation
lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)

res_cart_cv <- resample(loan_task, lrn_cart_cv, cv5, store_models = TRUE)


## --------------------------------------------------------------------------------------------------------
i=5
rpart::plotcp(res_cart_cv$learners[[i]]$model)




## --------------------------------------------------------------------------------------------------------
lrn_cart_cp <- lrn("classif.rpart", predict_type = "prob", cp = 0.011)

res <- benchmark(data.table(
  task       = list(loan_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_cart_cp),
  resampling = list(cv5)
), store_models = TRUE)


## --------------------------------------------------------------------------------------------------------
res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))


## --------------------------------------------------------------------------------------------------------
# Create a pipeline which encodes and then fits an XGBoost model
lrn_xgboost <- lrn("classif.xgboost", predict_type = "prob")
pl_xgb <- po("encode") %>>%
  po(lrn_xgboost)

lrn_svm <- lrn("classif.svm", predict_type = "prob")
pl_svm <- po("encode") %>>%
  po(lrn_svm)

lrn_ranger <- lrn("classif.ranger", predict_type = "prob")
pl_ranger <- po("encode") %>>%
  po(lrn_ranger)

# Now fit as normal ... we can just add it to our benchmark set
res <- benchmark(data.table(
  task       = list(loan_task),
  learner    = list(lrn_baseline,
                    pl_svm,
                    lrn_cart,
                    lrn_cart_cp,
                    pl_xgb,
                    pl_ranger
                    ),
  resampling = list(cv5)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))


## --------------------------------------------------------------------------------------------------------



## --------------------------------------------------------------------------------------------------------
as.data.table(lrn("classif.ranger")$param_set)[,
  .(id, class, lower, upper, nlevels)]




## --------------------------------------------------------------------------------------------------------
set.seed(123)

tuner=tnr("grid_search")
num_trees_range <- seq(from = 100, to = 500, by = 50)
alpha_range <- seq(from = 0, to = 1, by = 0.1)
learner= lrn("classif.ranger",
             
    num.trees = to_tune(num_trees_range),  # Number of trees to tune
    mtry = to_tune(4,7),        # Number of variables randomly sampled at each split
    #min.node.size = to_tune(2, 10), # Minimum node size for a split
    #splitrule = to_tune("gini"), # Splitting rule
    alpha = to_tune(alpha_range)
    #verbose=,
)
instance = ti(
  task = loan_task,
  learner,
  resampling = rsmp("cv", folds = 5),  # 3-fold cross-validation
  measures = msr("classif.ce"),       # Classification error
  terminator = trm("none")           # No early stopping
  
  )
tuner$optimize(instance)



## --------------------------------------------------------------------------------------------------------
result<-as.data.table(instance$archive,
  measures = msrs(c("classif.fpr", "classif.fnr")))[ ,
  .(num.trees,mtry,alpha,classif.ce, classif.fpr, classif.fnr)]%>%
  arrange(classif.ce)
result


## --------------------------------------------------------------------------------------------------------
#autoplot(instance, type = "surface")
#plot(result$alpha,result$classif.fpr)

lrn_ranger_tuned = lrn("classif.ranger")
lrn_ranger_tuned$param_set$values = instance$result_learner_param_vals

lrn_ranger_tuned$train(loan_task)$model


## --------------------------------------------------------------------------------------------------------
set.seed(123)
splits = partition(loan_task)

lrn_ranger = lrn("classif.ranger", predict_type = "prob", mtry=5,num.trees=300,alpha=0.7,num.threads=1)
lrn_ranger$train(loan_task, splits$train)
prediction = lrn_ranger$predict(loan_task, splits$test)
prediction



## --------------------------------------------------------------------------------------------------------

prediction$confusion
autoplot(prediction)
prediction$score(msr("classif.acc"))
prediction$score(msr("classif.auc"))
prediction$score(msr("classif.ce"))
prediction$score(msr("classif.fnr"))
prediction$score(msr("classif.fpr"))
autoplot(prediction, type = "roc")
autoplot(prediction, type = "prc")
autoplot(prediction, type = "threshold", measure = msr("classif.fnr"))
autoplot(prediction, type = "threshold", measure = msr("classif.acc"))


## --------------------------------------------------------------------------------------------------------
set.seed(123)
rr = resample(
  task = loan_task,
  learner = lrn("classif.ranger", predict_type = "prob", mtry=5,num.trees=300,alpha=0.7,num.threads=1),
  resampling = rsmp("cv", folds = 5)
)
autoplot(rr, type = "roc")
autoplot(rr, type = "prc")
autoplot(prediction, type = "threshold", measure = msr("classif.fnr"))
autoplot(prediction, type = "threshold", measure = msr("classif.acc"))


## --------------------------------------------------------------------------------------------------------
knitr::purl(input = "summative_classsification.Rmd", output = "report.R")

