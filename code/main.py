# Thomas Hepner
# 10/22/2016
# Analytics Vidya: MedCamp Hackathon

# Import libraries
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Set random seed
random.seed(0)

def load_merge_data():
    """ Load competition train and test data. Appropriately merge datasets and create Target outcome.
        Returns train and test data.
    """
    
    # Set work directory
    os.chdir('D:/OneDrive/Documents/AnalyticsVidya/data')        
    
    # Load data
    train = pd.read_csv("train.csv", sep = ",")
    test = pd.read_csv("Test.csv", sep = ",")
    camp_detail = pd.read_csv("Health_Camp_Detail.csv", sep = ",")
    patient_profile = pd.read_csv("Patient_Profile.csv", sep = ",")
    attended_health_camp_1 = pd.read_csv("First_Health_Camp_Attended.csv", sep = ",")
    attended_health_camp_2 = pd.read_csv("Second_Health_Camp_Attended.csv", sep = ",")
    attended_health_camp_3 = pd.read_csv("Third_Health_Camp_Attended.csv", sep = ",")
    
    # Merge data
    train = train.merge(camp_detail, how = 'left', on = 'Health_Camp_ID')
    train = train.merge(patient_profile, how = 'left', on = 'Patient_ID')    
    
    test = test.merge(camp_detail, how = 'left', on = 'Health_Camp_ID') 
    test = test.merge(patient_profile, how = 'left', on = 'Patient_ID')        
    
    # Add Target variable
    attended_health_camp_1 = attended_health_camp_1.drop(['Unnamed: 4'], axis = 1)
    attended_health_camp_1['Target'] = 1
    attended_health_camp_2['Target'] = 1       
    attended_health_camp_3['Target'] = [1 if x != 0 else 0 for x in attended_health_camp_3['Last_Stall_Visited_Number']] 
    
    # Reduce dimensions
    columns = ['Health_Camp_ID', 'Patient_ID', 'Target']
    attended_health_camp_1 = attended_health_camp_1[columns]
    attended_health_camp_2 = attended_health_camp_2[columns]
    attended_health_camp_3 = attended_health_camp_3[columns]
    
    # Concatenate data
    attended_all = pd.concat([attended_health_camp_1, attended_health_camp_2, attended_health_camp_3], axis = 0)
    
    train = train.merge(attended_all, how = 'left', on = ["Health_Camp_ID","Patient_ID"])
    train.ix[train['Target'].isnull(), 'Target'] = 0   
    train['Target'] = train['Target'].astype(np.int32)
 
    # Separate Target data
    y_train = train['Target']
    X_train =  train.drop('Target', axis = 1)
    X_test = test
    
    return X_train, X_test, y_train
    
def clean_data(X_train, X_test):
    """ Eliminates unnecessary features.
        Generates time-based features.
        Returns train and test data.
    """
    
    # Convert variables to Dates
    X_train['Registration_Date'] = pd.to_datetime(X_train['Registration_Date'])
    X_train['Camp_Start_Date'] = pd.to_datetime(X_train['Camp_Start_Date'])
    X_train['Camp_End_Date'] = pd.to_datetime(X_train['Camp_End_Date']) 
    X_train['First_Interaction'] = pd.to_datetime(X_train['First_Interaction'])
    
    X_test['Registration_Date'] = pd.to_datetime(X_test['Registration_Date'])
    X_test['Camp_Start_Date'] = pd.to_datetime(X_test['Camp_Start_Date'])
    X_test['Camp_End_Date'] = pd.to_datetime(X_test['Camp_End_Date'])  
    X_test['First_Interaction'] = pd.to_datetime(X_test['First_Interaction'])    
    
    # Create Duration Variables
    X_train['Camp_Duration'] = [int(i.days) for i in (X_train['Camp_End_Date'] - X_train['Camp_Start_Date'])] 
    X_test['Camp_Duration'] = [int(i.days) for i in (X_test['Camp_End_Date'] - X_test['Camp_Start_Date'])]   
    
    X_train['CampEnd_diff_Registration'] = ((X_train['Camp_End_Date'] - X_train['Registration_Date']).dt.days).astype(np.float32)
    X_test['CampEnd_diff_Registration'] = ((X_test['Camp_End_Date'] - X_test['Registration_Date']).dt.days).astype(np.float32)
    
    X_train['CampStart_diff_Registration'] = ((X_train['Camp_Start_Date'] - X_train['Registration_Date']).dt.days).astype(np.float32)
    X_test['CampStart_diff_Registration'] = ((X_test['Camp_Start_Date'] - X_test['Registration_Date']).dt.days).astype(np.float32)   
    
    X_train['CampEnd_diff_First'] = ((X_train['Camp_End_Date'] - X_train['First_Interaction']).dt.days).astype(np.float32)
    X_test['CampEnd_diff_First'] = ((X_test['Camp_End_Date'] - X_test['First_Interaction']).dt.days).astype(np.float32)      
    
    X_train['CampStart_diff_First'] = ((X_train['Camp_Start_Date'] - X_train['First_Interaction']).dt.days).astype(np.float32)
    X_test['CampStart_diff_First'] = ((X_test['Camp_Start_Date'] - X_test['First_Interaction']).dt.days).astype(np.float32)         
    
    X_train = X_train.drop(['Camp_Start_Date', 'Camp_End_Date'], axis = 1)
    X_test = X_test.drop(['Camp_Start_Date', 'Camp_End_Date'], axis = 1)    
        
    # Create Patient Response variables
    X_train['Patient_Response'] = ((X_train['Registration_Date'] - X_train['First_Interaction']).dt.days).astype(np.float32)
    X_test['Patient_Response'] = ((X_test['Registration_Date'] - X_test['First_Interaction']).dt.days).astype(np.float32)   
    
    X_train['Patient_Response_x_Camp_Duration'] = X_train['Patient_Response'] * X_train['Camp_Duration']
    X_test['Patient_Response_x_Camp_Duration'] = X_test['Patient_Response'] * X_test['Camp_Duration']
    
    X_train = X_train.drop(['Registration_Date', 'First_Interaction'], axis = 1)
    X_test = X_test.drop(['Registration_Date', 'First_Interaction'], axis = 1) 

    # Convert Education Score to float
    X_train.ix[X_train['Education_Score'] == 'None', 'Education_Score'] = None
    X_train['Education_Score'] = X_train['Education_Score'].astype(np.float32)
    X_test.ix[X_test['Education_Score'] == 'None', 'Education_Score'] = None
    X_test['Education_Score'] = X_test['Education_Score'].astype(np.float32)    
    
    # Convert age to float
    X_train.ix[X_train['Age'] == 'None', 'Age'] = None
    X_train['Age'] = X_train['Age'].astype(np.float32)
    X_test.ix[X_test['Age'] == 'None', 'Age'] = None
    X_test['Age'] = X_test['Age'].astype(np.float32)     
    
    # Remove ID variables
    columns = ['Patient_ID', 'Health_Camp_ID'] 
    X_train = X_train.drop(columns, axis = 1)
    submission = X_test[['Patient_ID', 'Health_Camp_ID']]
    X_test = X_test.drop(columns, axis = 1) 
    
    # Drop unnecessary variables
    X_train = X_train.drop(['Category3', 'Var3', 'Var4'], axis = 1)
    X_test = X_test.drop(['Category3', 'Var3', 'Var4'], axis = 1)
    
    # Fill null values
    X_train = X_train.fillna(value = -1000.0)        
    X_test = X_test.fillna(value = -1000.0) 
        
    return X_train, X_test, submission
    
def feature_engineering(X_train, X_test):
    """ Builds new features for model.
        Returns train and test data.
    """

    # Social Media
    X_train['Social_Media'] = ((X_train['Online_Follower'] == 1) | (X_train['Facebook_Shared'] == 1) | (X_train['Twitter_Shared'] == 1) | (X_train['LinkedIn_Shared'] == 1)).astype(np.int)
    X_test['Social_Media'] = ((X_test['Online_Follower'] == 1) | (X_test['Facebook_Shared'] == 1) | (X_test['Twitter_Shared'] == 1) | (X_test['LinkedIn_Shared'] == 1)).astype(np.int)
    drop_cols = ['Online_Follower', 'Facebook_Shared', 'Twitter_Shared', 'LinkedIn_Shared']
    X_train = X_train.drop(drop_cols, axis = 1)    
    X_test = X_test.drop(drop_cols, axis = 1)  

    # Interaction Variables
    X_train['Income_x_Education'] = X_train['Income'] * X_train['Education_Score']
    X_train['Income_x_Education'] = [-1000.0 if x < 0 else x for x in X_train['Income_x_Education']]
    X_train['Income_x_Education_x_Age'] = X_train['Income'] * X_train['Education_Score'] * X_train['Age']
    X_train['Income_x_Education_x_Age'] = [-1000.0 if x < 0 else x for x in X_train['Income_x_Education_x_Age']]

    X_test['Income_x_Education'] = X_test['Income'] * X_test['Education_Score']
    X_test['Income_x_Education'] = [-1000.0 if x < 0 else x for x in X_test['Income_x_Education']]
    X_test['Income_x_Education_x_Age'] = X_test['Income'] * X_test['Education_Score'] * X_test['Age']
    X_test['Income_x_Education_x_Age'] = [-1000.0 if x < 0 else x for x in X_test['Income_x_Education_x_Age']]
        
    return X_train, X_test    
    
def encode_labels(X_train, X_test, threshold): 
    """ Converts categorical features to integers. 
        Returns train and test data.
    """
    
    # Identify columns for label encoding
    columns = list(X_train.columns[X_train.dtypes == 'object'])
       
    # Return data if no columns identified   
    if(len(columns) == 0):
        return X_train, X_test
       
    # Transform columns
    for column in columns: 
        
        print "Column: " + str(column)
        # Filter data
        classes = X_train[column].unique()
        counts = X_train[column].value_counts()
        counts_classes = counts.index[counts <= threshold]
        
        # Set classes under threshold to 'identific'
        X_train.ix[X_train[column].isin(counts_classes), column] = 'identific'        
        X_test.ix[X_test[column].isin(counts_classes), column] = 'identific'          
        
        # Classes in test not in train sent to 'identific'
        X_test.ix[X_test[column].isin(classes) == False, column] = 'identific'
        
        # Perform label encoding
        le = LabelEncoder()
        le.fit(X_train[column])
        X_train[column] = le.transform(X_train[column]).astype(np.uint32)
        X_test[column] = le.transform(X_test[column]).astype(np.uint32)
        
    # Return data
    return X_train, X_test    

def create_folds(X_train, y_train, n_folds):
    """ Creates n-folds from train data for cross validation purposes.
    """

    # Create cv split
    kfolds = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 0)
    train_indices = []
    validation_indices = []
    for train_index, validation_index in kfolds.split(X_train, y_train):
        train_indices.append(train_index)
        validation_indices.append(validation_index)
        
    return train_indices, validation_indices

def plot_learning_curves(model_xgb):
    """ Takes XGBOOST model as input.
        Plots train and validation AUC for each iteration of model.
    """
    # Retrieve model performance metrics
    results = model_xgb.evals_result()
    epochs = len(results['validation_0']['auc'])
    x_axis = range(0, epochs)

    # Plot RMSE
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['auc'], label='Train')
    ax.plot(x_axis, results['validation_1']['auc'], label='Validation')
    ax.legend()
    plt.ylabel('AUC')
    plt.title('XGBoost AUC Performance')
    plt.show()
    
def build_xgboost_model(X_train, y_train, train_indices, validation_indices): # , sample_size
    """ Takes train data and train and validation fold indices as inputs.
        Builds XGBOOST model.
        Returns model, cross validation AUC score, and predictions.
    """    
    
    # Record running time
    start = time.time()

    # Define model and parameters 
    model = xgb.XGBClassifier()    
        
    xgb_params = {
                'objective': 'binary:logistic'
                , 'base_score': 0.5 
                , 'n_estimators': 40 
                , 'learning_rate': 0.1
                , 'max_depth': 4 
                , 'scale_pos_weight': 2
                , 'seed': 0
                }
    model.set_params(**xgb_params)       
        
    # Store predictions
    predictions = np.zeros(X_train.shape[0], dtype = np.float64)

    # Train model on validation sets
    for i in range(0, n_folds): 
        print "Fold " + str(i) + " :"
                            
        # Fit model on sampled data        
        model.fit(X_train.ix[train_indices[i]], y_train[train_indices[i]]
            , eval_set = [(X_train.ix[train_indices[i]], y_train[train_indices[i]]), (X_train.ix[validation_indices[i]], y_train[validation_indices[i]])]            
            , eval_metric = "auc" 
            , early_stopping_rounds = 5
            , verbose = True)             
            
        # Evaluate predictions over 5-folds
        predictions[validation_indices[i]] = [x[1] for x in model.predict_proba(X_train.ix[validation_indices[i]])]
        score = str(round(roc_auc_score(y_train[validation_indices[i]], predictions[validation_indices[i]]), 4))
        print "ROC AUC Score: " + score
        
        # Print learning curves
        plot_learning_curves(model)
        
    # Evaluate predictions over 5-folds
    cv_score = str(round(roc_auc_score(y_train, predictions), 4))
    print "ROC AUC Score: " + cv_score         
        
    # Print running time
    end = time.time()
    print "\nTime Elapsed: " + str(end - start) + " seconds"
    
    return model, cv_score, predictions

 ### Execute code! ###
if __name__ == '__main__':
    """ Builds predictive model and generates predictions.
    """    
    
    print "1. Build datasets..."
    X_train, X_test, y_train = load_merge_data()   
    print "Train Shape: " + str(X_train.shape)
    print "Test Shape: " + str(X_train.shape)
    
    print "2. Clean data..."
    X_train, X_test, submission = clean_data(X_train, X_test)
    
    print "3. Encode categorical data as numeric..."
    threshold = 0
    X_train, X_test = encode_labels(X_train, X_test, threshold)
    
    print "4. Feature Engineering..."
    X_train, X_test = feature_engineering(X_train, X_test)   
    
    print "5. Create validation framework..."
    n_folds = 5
    train_indices, validation_indices = create_folds(X_train, y_train, n_folds)
    
    print "6. Build model and cross validate it..."
    model, cv_score, predictions = build_xgboost_model(X_train, y_train, train_indices, validation_indices)
    
    print "7. Feature importances..."
    importances = model.feature_importances_
    columns = list(X_train.columns)
    feature_importances = pd.DataFrame()
    feature_importances['features'] = columns
    feature_importances['importances'] = importances
    feature_importances = feature_importances.sort_values(by = 'importances', ascending = False)    
    feature_importances = feature_importances.reset_index(drop = True)
    print feature_importances
    
    print "8. Generate predictions and write to CSV..."
    submission['Outcome'] = [x[1] for x in model.predict_proba(X_test)]
    os.chdir('D:/OneDrive/Documents/AnalyticsVidya/data/submissions') 
    submission.to_csv('submission_xgb_' + str(cv_score) + '.csv', index = False)    
    os.chdir('D:/OneDrive/Documents/AnalyticsVidya/data') 