from sklearn import metrics

def fragment_metrics(y_test, y_test_predicted):  # function that calculates metrics at fragment level
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_predicted)  # create ROC
    auc = metrics.auc(fpr, tpr) # calculate AUC
    
    print ('Fragment AUC = ', auc, '\n')
    
    
    plt.plot(fpr,tpr)  #plot roc curve
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    
    sensitivity = tpr             #calculate sensitivity array for all threshold values
    specificity = 1 - fpr         #calculate specificity array for all threshold values
    
    # optimum threshold is selected where sensitivity+specificity is maximum.  
    # The argmax returns the index of maximum sensitivity+specificity
    opt_threshold = thresholds[(sensitivity + specificity).argmax()]   
    print('Optimum threshold at fragment level:', opt_threshold, '\n')
    
    #Based on the Optimum threshold, 
    #turn the predicted probability by fragment to category values (0 is normal and 1 is pathologic)
    y_test_predicted_by_classes=np.where(y_test_predicted > opt_threshold, 1, 0)
    
    
    #Calculate confusion matrix at the optimum threshold
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_test_predicted_by_classes).ravel()
    
    print('Metrics at optimal threshold at fragment level:')
    print ('True negative = ', tn)
    print ('False positive = ', fp)
    print ('False negative = ', fn)
    print ('True positive = ', tp)
    
    print ('Sensitivity (recall, true positive rate) = ', tp/(tp+fn))
    print ('Precision = ', tp/(tp+fp))
    
    
    return opt_threshold, auc




def calc_metrics(y_test_predicted, y_test, fragment_opt_threshold):  # function that calculates metrics at patient level
    
    #Based on the Optimum threshold at fragment level, 
    #turn the predicted probability by fragment to category values (0 is normal and 1 is pathologic)
    y_test_predicted_by_classes=np.where(y_test_predicted > fragment_opt_threshold, 1, 0)
    
    #Build and array y1 that contains patient_ID, true category at fragment level and predicted value at fragment level
    y1=pd.DataFrame(data=None, index=None, columns=['patient_ID', 'true_value', 'predicted_value'], dtype=None, copy=False)
    
    y1.patient_ID=test_sample_IDs 
    y1.predicted_value=y_test_predicted_by_classes
    y1.true_value=y_test
    
    # Convert the fragment level array to patient level, and rename the column name so that it makes more sense
    results = y1.groupby(['patient_ID']).mean()
    results.rename(columns = {'predicted_value':'predicted_prob'}, inplace = True) 
    
    # 
    fpr, tpr, thresholds = metrics.roc_curve(results['true_value'], results['predicted_prob'])  #create ROC
    auc = metrics.auc(fpr, tpr) # calculate AUC
    print('Patient AUC = ', auc, '\n')
    
    plt.plot(fpr,tpr, '*-')  #plot ROC curve
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    
    sensitivity = tpr        #calculate sensitivity array for all threshold values
    specificity = 1 - fpr    #calculate specificity array for all threshold values
    
    # optimum threshold at patient level is selected where sensitivity+specificity is maximum.  
    # The argmax returns the index of maximum sensitivity+specificity
    opt_threshold = thresholds[(sensitivity + specificity).argmax()]
    
    print('Optimum threshold at patient level:', opt_threshold, '\n')
    
    # Add a column to the results dataframe that has the predicted category
    results['predicted_value'] = (results['predicted_prob'] > opt_threshold)*1.0
    #y_test_predicted_by_classes=np.where(y_test_predicted > opt_threshold, 1, 0)
    
    # Calculate the metrics in the confusion matrix
    tn, fp, fn, tp = metrics.confusion_matrix(results.true_value, results.predicted_value).ravel()
    
    print('Metrics at optimal threshold at patient level:')
    print ('True negative = ', tn)
    print ('False positive = ', fp)
    print ('False negative = ', fn)
    print ('True positive = ', tp)
    
    print ('Sensitivity (recall, true positive rate) = ', tp/(tp+fn))
    print ('False positive rate = ', fp/(fp+tn))
    print ('Precision = ', tp/(tp+fp))
    
    
    
    
#   If you want to tune the patient thresheld by hand, this is the code  
#     threshold=0.6

#     results['predicted_value'] = (results['predicted_prob']>=threshold)*1.0
    
#     tn, fp, fn, tp = metrics.confusion_matrix(results.true_value, results.predicted_value).ravel()
    
    
#     print('Metrics at optimal threshold at patient level:')
#     print ('True negative = ', tn)
#     print ('False positive = ', fp)
#     print ('False negative = ', fn)
#     print ('True positive = ', tp)
    
#     print ('Sensitivity (recall, true positive rate) = ', tp/(tp+fn))
#     print ('Precision = ', tp/(tp+fp))
    