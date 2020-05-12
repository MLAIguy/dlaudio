import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import *

#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.utils import to_categorical, plot_model
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.utils import shuffle
from matplotlib import pyplot
import random2
from tensorflow.keras.regularizers import l2
import tensorflow.keras
os.environ['CUDA_VISIBLE_DEVICES']='0'
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG16, InceptionV3


np.random.seed(1)
tf.random.set_seed(2)

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000):
        self.mode = mode
        self.nfilt=nfilt
        self.nfeat=nfeat
        self.nfft=nfft
        self.rate=rate
        self.step=int(rate)
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')


df = pd.read_csv('/work/02929/pz2339/maverick2/container-test/data.csv')
df['binary_label']=df['label']
df['binary_label'][(df[df['binary_label']!='Normal']).index]='Pathologic'

df_train, df_test= train_test_split_by_patients(df, seed=10)


classes = ['Normal', 'Pathologic']
class_dist = df_train.groupby(['binary_label'])['length'].sum()

n_samples=2*int(df['length'].sum())
prob_dist=class_dist/class_dist.sum()
choices=np.random.choice(class_dist.index,p=prob_dist)

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
#plt.show()
plt.savefig('dist.pdf')


class_dist = df_train.groupby(['label'])['length'].sum()

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.savefig('dist2.pdf')



config= Config(mode='conv')

if config.mode=='conv':
    X, y, train_sample_IDs, train_sample_labels, train_sample_audiofiles = build_all_feat_for_training(df_train, shifting=0.333 )
    #M=np.arange(0,X.shape[0])
    #np.random.shuffle(M)
    
    #X = X[M, :, :, :]
    #y = y[M]
    
    y_flat=y#np.argmax(y, axis=1)
    class_weight=compute_class_weight('balanced', 
                                 np.unique(y_flat),
                                 y_flat)
    input_shape=(X.shape[1], X.shape[2], 1)
    
    #test set
    X_test, y_test, test_sample_IDs, test_sample_labels, test_sample_audiofiles = build_for_testing(df_test)


tf.debugging.set_log_device_placement(True)
with tf.device('/job:localhost/replica:0/task:0/device:CPU:0'):
    X_resized_RGB = tf.image.resize_with_pad(X, 299, 299)
    X_resized_RGB = tf.image.grayscale_to_rgb(X_resized_RGB)

    X_test_resized_RGB = tf.image.resize_with_pad(X_test, 299, 299)
    X_test_resized_RGB = tf.image.grayscale_to_rgb(X_test_resized_RGB)



#original_image = tf.placeholder("float", [None, 99, 13, 1])
#new_resized_image = tf.image.resize_image_with_pad(original_image, 299, 299)
#new_resized_image_RGB = tf.image.grayscale_to_rgb(new_resized_image)

#with tf.Session() as session:
#    X_resized_RGB = session.run(new_resized_image_RGB, feed_dict={original_image: X})
#    print(new_resized_image_RGB)
#    X_test_resized_RGB = session.run(new_resized_image_RGB, feed_dict={original_image: X_test})
    
#    print(X_resized_RGB.shape)
#    print(X_test_resized_RGB.shape)


