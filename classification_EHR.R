require(cvTools); require(randomForest); require(caret); require(verification); require(tictoc)
require(e1071)
load("C:/Users/anisdavoudi/Dropbox (UFL)/idealist project drlirium/idealist workspace_2.RData")
#delirium_data<-delirium_cam[which(delirium_cam$encounter_deiden_id %in%cam_data_final$encounter_deiden_id),]
delirium_data<-Processed_data_delirium_cam_development_cohortt[which(Processed_data_delirium_cam_development_cohortt$encounter_deiden_id %in%
                                                                       cam_data_final$encounter_deiden_id),]
encounters<-delirium_data$encounter_deiden_id
delirium_data<-delirium_data[,-which(colnames(delirium_data) %in% Idealist_feature_list_modified$feature_name[which(Idealist_feature_list_modified$`excluded prior to feature selection`==1)])]
delirium_data$encounter_deiden_id<-encounters
delirium_data$label<-NA
for(i in 1:nrow(delirium_data)){
  delirium_data$label[i]<-delirium_cam$delirium_cam[delirium_cam$encounter_deiden_id==
                                                      delirium_data$encounter_deiden_id[i]]
}
for(i in 1:ncol(delirium_data)){
  print(i)
  print(colnames(delirium_data)[i])
  print(class(delirium_data[,i]))
  if(class(delirium_data[,i])=="character"){
    delirium_data[,i]<-as.factor(delirium_data[,i])
  }
}

data_new<-delirium_data
data_new<-as.data.frame(data_new)
data_new<-data_new[,-which(colnames(data_new)=="outcome")]
for(i in 1:(ncol(data_new)-1)){
  ii= which(Idealist_feature_list_modified$feature_name==colnames(data_new)[i])
  if(Idealist_feature_list_modified$feature_type_category[ii]=="Nominal"){
    data_new[,i]<-as.factor(data_new[,i])
  }
}

data_new$label<-as.factor(data_new$label)
data_new<-data_new[,-which(colnames(data_new)=="encounter_deiden_id")]
data_new$post_op_loc<-as.factor(data_new$post_op_loc)
run_rf_model_sed<-function(train,validation,colname)
{
  ### feature selection
  #train<-remove_colinear_variables(train)
  
  colnams<-colnames(train)
  validation<-validation[,which(colnames(validation) %in% colnams)]
  ind=which(colnames(train)=="label")
  #train<-upSample(train[,-ind], train[,ind], yname = "label")
  mean_t<-vector();sd_t<-vector()
  for(k in 1:(dim(train)[2]-1)){
    ii=which(Idealist_feature_list_modified$feature_name==colnames(train)[k]) 
    if(Idealist_feature_list_modified$feature_type_category[ii]=="Continuous"){
      mean_t[k]<-mean(train[,k], na.rm = TRUE)
      sd_t[k]<-sd(train[,k], na.rm = TRUE)
      train[,k]<-(train[,k]-mean_t[k])/sd_t[k]
    }
  }
  
  for(k in 1:(dim(validation)[2]-1)){
    ii=which(Idealist_feature_list_modified$feature_name==colnames(train)[k]) 
    if(Idealist_feature_list_modified$feature_type_category[ii]=="Continuous"){
      validation[,k]<-(validation[,k]-mean_t[k])/sd_t[k]
    }
  }
  
  ntree_values <- c(50,100,200,500) ; mtry_values <- c(1,2,5) 
  result <- NULL
  result <- data.frame(matrix(ncol=4)) ; colnames(result) <- c("auc","acc","numoftrees","mtry")  
  for(g in 1:length(ntree_values)){
    for(h in 1:length(mtry_values)){
      ntree = ntree_values[g] ; mtry = mtry_values[h]
      model = randomForest(x=train[,-ncol(train)],y=train[,ncol(train)],replace=TRUE,ntree=ntree,mtry=mtry) 
      outcome_pred=predict(model,validation[,-ind],type="response")
      a=confusionMatrix(outcome_pred,validation[,ind])
      mod_result <- data.frame(matrix(ncol=4)) ; colnames(mod_result) <- c("auc","acc","numoftrees","mtry")
      mod_result$acc=a$overall[1]
      mod_result$auc=auc(as.numeric(validation[,ind])-1,as.numeric(outcome_pred)-1)
      mod_result$numoftrees=ntree;
      mod_result$mtry=mtry;
      result=rbind(result,mod_result)
    }
  }
  result=na.omit(result);
  ntree=min(result[result$auc==max(result$auc,na.rm = TRUE),"numoftrees"])
  mtry=min(result[result$auc==max(result$auc,na.rm = TRUE),"mtry"])
  final_model=randomForest(x=train[,-ncol(train)],y=train[,ncol(train)],replace=TRUE,ntree=ntree,mtry=mtry,importance = TRUE)
  outcome_pred=predict(final_model,validation[,-ncol(validation)],type="response")
  mteric_final_model=calculate_report_sed(outcome_pred,validation[,ncol(validation)])
  return (list("model"=final_model,"acc"=result[result$auc==max(result$auc,na.rm = TRUE),"acc"],"metric"=mteric_final_model$metric, "ntree"=ntree, "mtry"=mtry,"auc"=result[result$auc==max(result$auc,na.rm = TRUE),"auc"]))
}


predict_label_rf_sed<-function(data_new,colname)
{
  data_new<-cbind(data_new[,-which(colnames(data_new)=="label")], data_new[,which(colnames(data_new)=="label")])
  colnames(data_new)[ncol(data_new)]<-"label"
  un<-unique(data_new$id)
  proc_data<-data_new
  models_rf <- vector(mode = "list", length = 5)
  acc <- vector(mode = "list", length = 5)
  auc <- vector(mode = "list", length = 5)
  acc_test <- vector(mode = "list", length = 5)
  auc_test <- vector(mode = "list", length = 5)
  important_features <- vector(mode = "list", length = 5)
  complete_model<-vector(mode = "list", length = 5)
  complete_model_on_test_results<-data.frame(matrix(ncol=18));
  colnames(complete_model_on_test_results)<-c("auc","acc", "acc_l", "acc_h", "NIR", "p_val","kappa", "Sensitivity_1", "Specificity_1", "PPV_1", "NPV_1",
                                              "Precision_1","Recall_1","F1_1","Prevalence_1","DetectionRate_1", "DetectionPrevalence_1", "BalancedAccuracy_1")
  results <- list()
  metric_allmodel <- data.frame(matrix(ncol=18)) ; colnames(metric_allmodel) <- colnames(complete_model_on_test_results)
  metric_model_this_run<-data.frame(matrix(ncol=18)); colnames(metric_model_this_run) <- colnames(complete_model_on_test_results)
  
  count <- 1;
  k <- 5
  #folds_1 <- cvFolds(NROW(proc_data), K=k)
  #to chosse by unique participant id
  folds_1<-cvFolds(nrow(proc_data), K=k)
  actual<-vector(); predicted<-vector()
  for(j in 1:5){     
    test_data<-proc_data[folds_1$subsets[which(folds_1$which == j)],] 
    develop_data<-proc_data[folds_1$subsets[-which(folds_1$which == j)],]
    print("j=");print(j);
    new_data <- develop_data
    new_data <- new_data[sample(nrow(new_data)),]
    k <- 5 
    folds <- cvFolds(NROW(new_data), K=k)
    
    
    for(f in 1:k){
      print(f)
      train <- new_data[folds$subsets[which(folds$which != f)],] #Set the training set
      validation <- new_data[folds$subsets[which(folds$which == f)],] #Set the validation set
      newpred_rf<-run_rf_model_sed(train, validation, colname)
      models_rf[[count]]<-newpred_rf$model
      important_features[[count]]<-newpred_rf$model$importance;
      metric_allmodel=rbind(metric_allmodel,newpred_rf$metric)
      metric_model_this_run=rbind(metric_model_this_run, newpred_rf$metric)
      acc[[count]]=newpred_rf$acc[1]
      auc[[count]]=newpred_rf$auc[1]
      acc_test[[j]]=newpred_rf$acc[1]
      auc_test[[j]]=newpred_rf$auc[1]
      count=count+1;
    }
    
    temp_count<-which.max(unlist(auc_test))
    best_model=models_rf[temp_count]
    temp<-metric_model_this_run
    
    ind=which(colnames(test_data)=="label")
    actual_outcome_test<-(test_data[,ncol(test_data)])
    mean_d<-vector();sd_d<-vector()

    for(k in 1:(dim(develop_data)[2]-1)){
      ii=which(Idealist_feature_list_modified$feature_name==colnames(develop_data)[k]) 
      if(Idealist_feature_list_modified$feature_type_category[ii]=="Continuous"){
        mean_d[k]<-mean(develop_data[,k], na.rm = TRUE)
        sd_d[k]<-sd(develop_data[,k], na.rm = TRUE)
        develop_data[,k]<-(develop_data[,k]-mean_d[k])/sd_d[k]
      }
    }
    for(k in 1:(ncol(test_data)-1)){
      ii=which(Idealist_feature_list_modified$feature_name==colnames(test_data)[k]) 
      if(Idealist_feature_list_modified$feature_type_category[ii]=="Continuous"){
        test_data[,k]<-(test_data[,k]-mean_t[k])/sd_t[k]
      }
    }
    
    #develop_data<-remove_colinear_variables(develop_data)
    colnams<-colnames(develop_data)
    test_data<-test_data[,which(colnames(test_data) %in% colnams)]
    
    #develop_data<-remove_colinear_variables(develop_data)
    complete_model[[j]]<-randomForest(x=develop_data[,-ncol(develop_data)],y=develop_data[,ncol(develop_data)], replace=TRUE, ntree=newpred_rf$ntree, mtry=newpred_rf$mtry)
    complete_model_on_test<-predict(complete_model[[j]], as.data.frame(test_data[,-ncol(test_data)]), type = "response")
    temping<-calculate_report_sed(complete_model_on_test,actual_outcome_test)$metric
    complete_model_on_test_results<-rbind(complete_model_on_test_results,temping)
    actual<-c(actual, actual_outcome_test)
    predicted<-c(predicted, complete_model_on_test)
  }
  
  proc_data<-proc_data[,which(colnames(proc_data) %in% colnams)]
  best_model_rf=models_rf[which.max(unlist(auc))]
  imp_feature_rf=important_features[which.max(unlist(auc))]
  ind=which(colnames(proc_data)=="label")
  actual_outcome=proc_data[,ind]
  #best_model_output_pred=predict(best_model_rf[[1]],as.data.frame(proc_data[,-ind]),type="response")
  #best_model_metric=calculate_report_sed(best_model_output_pred,actual_outcome)$metric
  
  return(list("best_model"=best_model_rf,"metric_allmodel"=metric_allmodel,"imp_feature"=imp_feature_rf,
              "complete_model"=complete_model, "complete_model_on_test"=complete_model_on_test, "complete_model_on_test_results"=complete_model_on_test_results,
              "predicted"=predicted, "actual"=actual))
}
require(cvTools); require(randomForest); require(caret); require(verification); require(tictoc)




calculate_report_sed<-function(predicted, reported){
  d<-confusionMatrix(predicted, reported)
  auc<-auc(as.numeric(reported)-1, as.numeric(predicted)-1)
  acc<-d$overall[1]
  acc_l<-d$overall[3]
  acc_h<-d$overall[4]
  NIR<-d$overall[5]
  p_val<-d$overall[6]
  kappa<-d$overall[2]
  
  i=1
  for(i in 1:1){
    assign(paste0("Sensitivity_",i),d$byClass[1])
    assign(paste0("Specificity_",i),d$byClass[2])
    assign(paste0("PPV_",i),d$byClass[3])
    assign(paste0("NPV_",i),d$byClass[4])
    assign(paste0("Precision_",i),d$byClass[5])
    assign(paste0("Recall_",i),d$byClass[6])
    assign(paste0("F1_",i),d$byClass[7])
    assign(paste0("Prevalence_",i),d$byClass[8])
    assign(paste0("DetectionRate_",i),d$byClass[9])
    assign(paste0("DetectionPrevalence_",i),d$byClass[10])
    assign(paste0("BalancedAccuracy_",i),d$byClass[11])
  }
  metric<-data.frame(auc,acc, acc_l, acc_h, NIR, p_val,kappa, Sensitivity_1, Specificity_1, PPV_1, NPV_1,
                     Precision_1,Recall_1,F1_1,Prevalence_1,DetectionRate_1, DetectionPrevalence_1, BalancedAccuracy_1)
  return (list("metric"=metric))
}


set.seed(1)
tic()
rf_results_16sec_sed_alldatafortest<-predict_label_rf_sed(data_new, label)
toc()

rf_results<-rf_results_16sec_sed_alldatafortest
save(file = "C:/Users/anisdavoudi/Dropbox (UFL)/idealist project drlirium/rf_results.rda", 
     rf_results)
