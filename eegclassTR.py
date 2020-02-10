

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

#data = scipy.io.loadmat('C:/Users/Hamidreza/Desktop/new-research/EEG-new work/FBCSP/TRCSP/features_aw_48_0.mat')
#X_t1=data['features']
#data = scipy.io.loadmat('C:/Users/Hamidreza/Desktop/new-research/EEG-new work/FBCSP/TRCSP/features_aw_812_0.mat')
#X_t2=data['features']
#data = scipy.io.loadmat('C:/Users/Hamidreza/Desktop/new-research/EEG-new work/FBCSP/TRCSP/features_aw_1216_0.mat')
#X_t3=data['features']
#data = scipy.io.loadmat('C:/Users/Hamidreza/Desktop/new-research/EEG-new work/FBCSP/TRCSP/features_aw_1620_0.mat')
#X_t4=data['features']
#data = scipy.io.loadmat('C:/Users/Hamidreza/Desktop/new-research/EEG-new work/FBCSP/TRCSP/features_aw_2024_0.mat')
#X_t5=data['features']
#data = scipy.io.loadmat('C:/Users/Hamidreza/Desktop/new-research/EEG-new work/FBCSP/TRCSP/features_aw_2428_0.mat')
#X_t6=data['features']
#data = scipy.io.loadmat('C:/Users/Hamidreza/Desktop/new-research/EEG-new work/FBCSP/TRCSP/features_aw_2428_0.mat')
#X_t7=data['features']
#data = scipy.io.loadmat('C:/Users/Hamidreza/Desktop/new-research/EEG-new work/FBCSP/TRCSP/features_aw_2832_0.mat')
#X_t8=data['features']
#data = scipy.io.loadmat('C:/Users/Hamidreza/Desktop/new-research/EEG-new work/FBCSP/TRCSP/features_aw_3236_0.mat')
#X_t9=data['features']
#data = scipy.io.loadmat('C:/Users/Hamidreza/Desktop/new-research/EEG-new work/FBCSP/TRCSP/features_aw_3640_0.mat')
#X_t10=data['features']
#X_t=np.concatenate((X_t1,X_t2,X_t3,X_t4,X_t5,X_t6,X_t7,X_t8,X_t9,X_t10),axis=0)
#
data = scipy.io.loadmat('C:/Users/Hamidreza/Desktop/new-research/EEG-new work/FBCSP/TRCSP/features_av_440_0.mat')
X_t=data['features']
X_t= np.transpose(X_t, (2, 0, 1))

M=84
X_train=X_t[:M,:,:]
X_test=X_t[M:,:,:]

plt.figure(figsize=[8,6])
plt.imshow(X_t[6,:,:])
plt.show() 

label=scipy.io.loadmat("C:/Users/Hamidreza/Desktop/new-research/EEG-new work/true_labels_av.mat")
y_t =label['true_y'][0]-1
y_t=y_t.astype(np.float64)

y_train=y_t[:M]
y_test=y_t[M:]

from sklearn import metrics
from sklearn.metrics import confusion_matrix
n_samples = 6*201   
dataa = X_train.reshape([M,n_samples, -1])
dataa_test = X_test.reshape([280-M,n_samples, -1])
dataa=np.squeeze(dataa) 
dataa_test=np.squeeze(dataa_test) 
##########################
from sklearn.naive_bayes import GaussianNB
expected=y_test
classifier=GaussianNB()
classifier.fit(dataa, y_train)
predicted = classifier.predict(dataa_test)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
#########################################################33
from sklearn.tree import DecisionTreeClassifier

clf_entropy = DecisionTreeClassifier(criterion = "entropy")
clf_entropy.fit(dataa, y_train)
pred2 = clf_entropy.predict(dataa_test)

print("Classification report for classifier %s:\n%s\n"
      % (clf_entropy, metrics.classification_report(expected, pred2)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, pred2))
###############################################
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf3 = LinearDiscriminantAnalysis( solver='svd')
clf3.fit(dataa, y_train)

predi = clf3.predict(dataa_test)
print("Classification report for classifier %s:\n%s\n"
      % (clf3, metrics.classification_report(expected, predi)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predi))
##########################################
from sklearn.neural_network import MLPClassifier                    
clff=MLPClassifier(activation='relu', alpha=1e-05, hidden_layer_sizes=(400, 200), batch_size=8,
       learning_rate_init=0.001, max_iter=300, random_state=1, solver='adam')
clff.fit(dataa, y_train)
predicted3 = clff.predict(dataa_test)

print("Classification report for classifier %s:\n%s\n"
      % (clff, metrics.classification_report(expected, predicted3)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted3))
#####################################
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(dataa, y_train)
predictions1 = model.predict(dataa_test)
print("Classification report for classifier %s:\n%s\n"
  % (model, metrics.classification_report(y_test, predictions1)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictions1)) 


#m=0
#
#kf=StratifiedKFold(5, random_state=None, shuffle=False)
#kf.get_n_splits(X_t, y_t)
#k=0
#for train_index, test_index in kf.split(X_t, y_t):
#    X_train, X_test = X_t[train_index], X_t[test_index]
#    y_train, y_test = y_t[train_index], y_t[test_index]
#    
#    if k==m:
#       break 
#    k=k+1
    
    
Y_train = np_utils.to_categorical(y_train, 2)
Y_test = np_utils.to_categorical(y_test, 2)

#########################################################################################################



#VVr = np.reshape(X_train, ( X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
VV=np.stack([X_train]*3, axis=-1)
#VVVr=np.concatenate((VVr,VVr), axis=3) 
#VV=np.concatenate((VVVr,VVr), axis=3)
#VVr1 = np.reshape(X_test, ( X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
VV1=np.stack([X_test]*3, axis=-1)

#VVVr1=np.concatenate((VVr1,VVr1), axis=3)
#VV1=np.concatenate((VVVr1,VVr1), axis=3)
input_shape=VV.shape[1:4]
#################################################VGG
## create the base pre-trained model
#base_model =  VGG16(weights='imagenet', include_top=False)
#
## add a global spatial average pooling layer
#x = base_model.output
## add a global spatial Max pooling layer
#x = GlobalAveragePooling2D()(x)
##x = Flatten()(x)
### let's add a fully-connected layer
##x1 = Dense(128, activation='relu')(x)
##x1 = normalization.BatchNormalization()(x1)
##x2 = Dense(64, activation='relu')(x1)
##x2 = normalization.BatchNormalization()(x2)
#predictions = Dense(2, activation='softmax')(x)
#
## this is the model we will train
#model = Model(inputs=base_model.input, outputs=predictions)
#
#for i, layer in enumerate(base_model.layers):
#   print(i, layer.name)
#for layer in model.layers[:19]:########VGG16
#   layer.trainable = False
#for layer in model.layers[19:]:
#   layer.trainable = True
######################################################
def createModel():
    model = Sequential()
    model.add(Conv2D(4, (3, 3), activation='relu', kernel_initializer=initializers.TruncatedNormal(stddev=0.01), bias_initializer=initializers.Constant(value=0.01), padding='same', input_shape=input_shape))
#    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Conv2D(8, (3, 3), activation='relu', kernel_initializer=initializers.TruncatedNormal(stddev=0.01), bias_initializer=initializers.Constant(value=0.01), padding='same'))
#    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(GlobalMaxPooling2D())
#    model.add(Flatten())
#    model.add(Dense(100, kernel_initializer=initializers.TruncatedNormal(stddev=1), bias_initializer=initializers.Constant(value=0.001)))
#    model.add(BatchNormalization())
#    model.add(Activation('relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(200, kernel_initializer=initializers.TruncatedNormal(stddev=1), bias_initializer=initializers.Constant(value=0.001)))
#    model.add(BatchNormalization())
#    model.add(Activation('relu'))
#    model.add(Dropout(0.8))
#    model.add(Dense(100, kernel_initializer=initializers.TruncatedNormal(stddev=1), bias_initializer=initializers.Constant(value=0.001)))
#    model.add(BatchNormalization())
#    model.add(Activation('relu'))
#    model.add(Dropout(0.8))
#    model.add(Dense(50, kernel_initializer=initializers.TruncatedNormal(stddev=1), bias_initializer=initializers.Constant(value=0.001)))
#    model.add(BatchNormalization())
#    model.add(Activation('relu'))
#    model.add(Dropout(0.9))
    model.add(Dense(2, activation='softmax'))
    
    return model
    

model = createModel()


model.summary()
batch_size = 8
epochs =200   
#opta=optimizers.SGD(lr=0.0005, momentum=0.9)
optadam=optimizers.Adam(lr=0.0001)
model.compile(optimizer=optadam, loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.8, patience=150, min_lr=0.0001)

history = model.fit(VV, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, 
                   validation_data=(VV1, Y_test), callbacks=[reduce_lr])
#model.save('AY.h5')

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
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Test Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)

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


