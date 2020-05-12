
def train_test_split_by_patients(df, seed=4):
    
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    
    for c in df['label'].unique():
        
        #IDs_in_this_class = df[df['label']==c]['ID'].unique()
        #IDs_in_this_class = shuffle(IDs_in_this_class)
        
        #n_samples_each_class = int(len(IDs_in_this_class)*0.1)
        
        #IDs_for_testing = shuffle(IDs_in_this_class)
        
        
        df_sub=df[df['label']==c]
        patients = df_sub['ID'].unique()
        n_patients_for_testing = int(len(patients)*0.1)
        #random2.seed(seed)
        patients=shuffle(patients, random_state=seed)
        patients_for_testing=patients[0:n_patients_for_testing]
        patients_for_training=patients[n_patients_for_testing:]
        
        df_sub_test = df_sub[df_sub['ID'].isin(patients_for_testing)]
        df_sub_train = df_sub[df_sub['ID'].isin(patients_for_training)]
        

        df_train = pd.concat([df_train, df_sub_train])
        df_test = pd.concat([df_test, df_sub_test])
    
    df_train = df_train.set_index('index').reset_index()#.drop(columns='level_0')
    df_test = df_test.set_index('index').reset_index()#.drop(columns='level_0')
    
#    n_new_patients = int((df.drop_duplicates('ID')).shape[0]*0.05)
    
#    n_test_existing_patients = int(df.shape[0]*0.05)

    return df_train, df_test
    