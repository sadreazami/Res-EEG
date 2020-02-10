  
from keras.models import Model
from keras.layers import Input, Dense, merge, Activation
from keras.utils import np_utils
import numpy as np
import keras 
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from keras.callbacks import ReduceLROnPlateau

from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, normalization
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation 
from keras import initializers, optimizers

data = scipy.io.loadmat('C:/Users/Hamidreza/Desktop/new-research/EEG-new work/FBCSP/TRCSP/features_al_440_0.mat')
X_t=data['features']
X_t= np.transpose(X_t, (2, 0, 1))

M=224
X_train=X_t[:M,:,:]
X_test=X_t[M:,:,:]

#plt.figure(figsize=[8,6])
#plt.imshow(X_t[6,:,:])
#plt.show() 

label=scipy.io.loadmat("C:/Users/Hamidreza/Desktop/new-research/EEG-new work/true_labels_al.mat")
y_t =label['true_y'][0]-1
y_t=y_t.astype(np.float64)

y_train=y_t[:M]
y_test=y_t[M:]

Y_train = np_utils.to_categorical(y_train, 2)
Y_test = np_utils.to_categorical(y_test, 2)

VV=np.stack([X_train]*3, axis=-1)
VV1=np.stack([X_test]*3, axis=-1)

input_shape=VV.shape[1:4]

np.random.seed(813306)
 
def build_resnet(input_shape, n_feature_maps, nb_classes):
    print ('build conv_x')
    x = Input(shape=(input_shape))
    conv_x = keras.layers.normalization.BatchNormalization()(x)
    conv_x = keras.layers.Conv2D(n_feature_maps, (3, 3), kernel_initializer=initializers.TruncatedNormal(stddev=0.01), bias_initializer=initializers.Constant(value=0.01), padding='same')(x)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
     
    print ('build conv_y')
    conv_z = keras.layers.Conv2D(n_feature_maps, (3, 3), kernel_initializer=initializers.TruncatedNormal(stddev=0.01), bias_initializer=initializers.Constant(value=0.01), padding='same')(conv_x)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)
     
    is_expand_channels = not (input_shape[-1] == n_feature_maps)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps, (3, 3), kernel_initializer=initializers.TruncatedNormal(stddev=0.01), bias_initializer=initializers.Constant(value=0.01), padding='same')(x)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.normalization.BatchNormalization()(x)
    print ('Merging skip connection')
    y = merge([shortcut_y, conv_z], mode='sum')
    y = Activation('relu')(y)
     
    print ('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv2D(n_feature_maps*2, (3, 3), kernel_initializer=initializers.TruncatedNormal(stddev=0.01), bias_initializer=initializers.Constant(value=0.01), padding='same')(x1)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
     
    print ('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps*2, (3, 3), kernel_initializer=initializers.TruncatedNormal(stddev=0.01), bias_initializer=initializers.Constant(value=0.01), padding='same')(conv_x)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps*2, (3, 3),kernel_initializer=initializers.TruncatedNormal(stddev=0.01), bias_initializer=initializers.Constant(value=0.01), padding='same')(x1)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.normalization.BatchNormalization()(x1)
    print ('Merging skip connection')
    y = merge([shortcut_y, conv_z], mode='sum')
    y = Activation('relu')(y)
     
    full = keras.layers.pooling.GlobalMaxPooling2D()(y)   
    out = Dense(nb_classes, activation='softmax')(full)
    print ('        -- model was built.')
    return x, out
    

epochs = 200
batch_size = 4  
    
nb_classes = len(np.unique(y_test))

x , y = build_resnet(VV.shape[1:], 2, nb_classes)
model = Model(input=x, output=y)
  
model.summary()
optadam=optimizers.Adam(lr=0.0001)
model.compile(optimizer=optadam, loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.8, patience=200, min_lr=0.0001)

history = model.fit(VV, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, 
                   validation_data=(VV1, Y_test), callbacks=[reduce_lr])


model.evaluate(VV1, Y_test, verbose=0) 
y_pred = model.predict(VV1)
y_pred=np.squeeze(y_pred)          

preds = np.argmax(y_pred, axis=1)    

# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Test Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

loss1=history.history['loss']
scipy.io.savemat('C:/Users/Hamidreza/Desktop/ISCAS 2019/loss1.mat', mdict={'loss1': loss1}) 
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Test Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
#
#acc1=history.history['acc']
#scipy.io.savemat('C:/Users/Hamidreza/Desktop/ISCAS 2019/acc1.mat', mdict={'acc1': acc1})


labels = {1:'Left', 2:'Right'}
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(preds, y_test,
                            target_names=[l for l in labels.values()]))

conf_mat = confusion_matrix(preds, y_test)

fig = plt.figure(figsize=(2,2))
width = np.shape(conf_mat)[1]
height = np.shape(conf_mat)[0]

res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
for i, row in enumerate(conf_mat):
    for j, c in enumerate(row):
        if c>0:
            plt.text(j-.2, i+.1, c, fontsize=16)
            
#cb = fig.colorbar(res)
plt.title('Confusion Matrix')
_ = plt.xticks(range(2), [l for l in labels.values()], rotation=90)
_ = plt.yticks(range(2), [l for l in labels.values()])