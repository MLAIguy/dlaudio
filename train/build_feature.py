

def build_all_feat_for_training(df, shifting=0.333):
    X=[]
    y=[]
    sample_IDs=[]
    sample_labels=[]
    sample_audiofiles=[]
    
    _min, _max=float('inf'), -float('inf')
    
    for i in range(0,df.shape[0]):
        
        ID, label, audiofile, binary_label=df.iloc[i][['ID', 'label', 'filename', 'binary_label']]

        path='/work/02929/pz2339/maverick2/container-test/clean/'+label+'/'+ID+'/'+audiofile
        rate, wav=wavfile.read(path)
        if wav.shape[0]<config.step:
            continue
        
        for audio_index in range(0, wav.shape[0]-config.step, int(config.step*shifting)):  #config.step*0.333
        #for audio_index in range(0, wav.shape[0]-config.step, config.step):
            sample = wav[audio_index:audio_index+config.step]
            X_sample=mfcc(sample, rate,
                     numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
            _min=min(np.amin(X_sample), _min)
            _max=max(np.amax(X_sample), _max)
            X.append(X_sample)
            y.append(classes.index(binary_label))
            sample_IDs.append(ID)
            sample_labels.append(label)
            sample_audiofiles.append(audiofile)
            config.min=_min
            config.max=_max

            
            
    X, y, sample_IDs, sample_labels, sample_audiofiles = np.array(X), np.array(y), np.array(sample_IDs), np.array(sample_labels), np.array(sample_audiofiles)
    X=(X-_min)/(_max-_min)
            
    if config.mode=='conv':
        X=X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode=='time':
        X=X.reshape(X.shape[0], X.shape[1], X.shape[2])
    #y=to_categorical(y, num_classes=2)
    
    #config.data=(X,y)
    
    return X, y, sample_IDs, sample_labels, sample_audiofiles

