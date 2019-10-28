"""
@author : Julie Bogoin
Master 2 BIB - 2019 2020
Projet Deep Learning
"""

#Import des modules
import numpy as np 
import os
import random
import keras
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import tensorflow as tf

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution3D, MaxPooling3D, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as k
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from math import sqrt

###### FONCTIONS ######

def creation_control (nbre_individu):
    #Génération des listes contenant nbre_individu poches choisies au hasard 
    echantillon_control = random.sample(control_list, nbre_individu)
    random.shuffle(echantillon_control)
    return echantillon_control

def creation_heme (nbre_individu):
    echantillon_heme = random.sample(heme_list, nbre_individu)
    random.shuffle(echantillon_heme)
    return echantillon_heme

def creation_nucleotide (nbre_individu): 
    echantillon_nucleotide = random.sample(nucleotide_list, nbre_individu)  
    random.shuffle(echantillon_nucleotide)
    return echantillon_nucleotide

def creation_steroid (nbre_individu):
    echantillon_steroid = random.sample(steroid_list, nbre_individu)
    random.shuffle(echantillon_steroid)
    return echantillon_steroid

def generation_voxels (voxels_tot, heme, nucleotide, control):
    #Génération de la liste des voxels
    echantillon_voxel = []
    for voxel in voxels_tot:
        if voxel.replace('.npy','') in heme:
            echantillon_voxel.append(voxel)
        if voxel.replace('.npy','') in nucleotide:
            echantillon_voxel.append(voxel)
        if voxel.replace('.npy','') in control:
            echantillon_voxel.append(voxel)
    random.shuffle(echantillon_voxel)
    return echantillon_voxel

def definition_de_x(path, echantillon_voxel):
    X = [np.load("{}/{}".format(path, voxel)) for voxel in echantillon_voxel]
    X = [np.squeeze(array) for array in X]
    X = np.array(X)
    return X

def definition_de_y(echantillon_voxel, nucleotide, heme, control):
    Y = []
    for voxel in echantillon_voxel:
        if voxel.replace('.npy','') in nucleotide:
            Y.append(0)
        elif voxel.replace('.npy','') in heme:
            Y.append(1)
        elif voxel.replace('.npy','') in control:
            Y.append(2)    
    Y  = np.array(Y)
    return Y

#Modele de DeepDrug3D
def creation_modele ():
    model = Sequential()
    # Conv layer 1
    model.add(Convolution3D(
        input_shape = (14,32,32,32),
        filters=64,
        kernel_size=5,
        padding='valid',     # Padding method
        data_format='channels_first',
    ))
    model.add(LeakyReLU(alpha = 0.1))
    # Dropout 1
    model.add(Dropout(0.2))
    # Conv layer 2
    model.add(Convolution3D(
        filters=64,
        kernel_size=3,
        padding='valid',     # Padding method
        data_format='channels_first',
    ))
    model.add(LeakyReLU(alpha = 0.1))
    # Maxpooling 1
    model.add(MaxPooling3D(
        pool_size=(2,2,2),
        strides=None,
        padding='valid',    # Padding method
        data_format='channels_first'
    ))
    # Dropout 2
    model.add(Dropout(0.4))
    # FC 1
    model.add(Flatten())
    model.add(Dense(128)) 
    model.add(LeakyReLU(alpha = 0.1))
    # Dropout 3
    model.add(Dropout(0.4))
    # Fully connected layer 2 to shape (3) for 3 classes
    model.add(Dense(3))
    model.add(Activation('softmax'))
    return model   


###### MAIN ######

        #Génération des listes poches
control_list = []
heme_list = []
steroid_list = []
nucleotide_list = []

with open ("./Data/control.list", "r") as fillin:
    for ligne in fillin:
        control = ligne.replace('\n','')
        control_list.append(control)
        
with open ("./Data/heme.list", "r") as fillin:
    for ligne in fillin:
        heme = ligne.replace('\n','')
        heme_list.append(heme)

with open ("./Data/steroid.list", "r") as fillin:
    for ligne in fillin:
        steroid = ligne.replace('\n','')
        steroid_list.append(steroid)

with open ("./Data/nucleotide.list", "r") as fillin:
    for ligne in fillin:
        nucleotide = ligne.replace('\n','')
        nucleotide_list.append(nucleotide)

        #Génération de la liste Voxels
voxels_path = "./Data/Voxels/"
voxels_tot = os.listdir(voxels_path)
voxels_tot = list(filter(lambda fichier: fichier.endswith('.npy'), voxels_tot))

    ###### JEU D'APPRENTISSAGE ######
train_control = creation_control(50)
train_heme = creation_heme(50)
train_nucleotide = creation_nucleotide(50)
train_voxels = generation_voxels(voxels_tot, train_heme, train_nucleotide, train_control)

X_train = definition_de_x(voxels_path, train_voxels)
print("\nFormat du X_train: {}".format(X_train.shape))
Y_train = definition_de_y(train_voxels, train_nucleotide, train_heme, train_control)
encoded_Y_train = to_categorical(Y_train)
print("Format du Y_train: {}\n".format(encoded_Y_train.shape))

    ###### JEU DE TEST ######
test_control = creation_control(130)
test_heme = creation_heme(130)
test_nucleotide = creation_nucleotide(130)
test_voxels = generation_voxels(voxels_tot, test_heme, test_nucleotide, test_control)

X_test = definition_de_x(voxels_path, test_voxels)
Y_test = definition_de_y(test_voxels, test_nucleotide, test_heme, test_control)
encoded_Y_test = to_categorical(Y_test)

    ###### CREATION DU MODEL ######
critor = EarlyStopping(monitor = "val_loss", patience = 3, mode = "min")
my_model = creation_modele()
my_model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])

history = my_model.fit(X_train, encoded_Y_train, epochs = 50, batch_size = 20,
             validation_split = 0.1, callbacks = [critor])

        #Sauvegarder le model
my_model.save('./my_model.h5')

    ###### EVALUATION DU MODEL ######
evaluation = my_model.evaluate(X_test, encoded_Y_test)
print("\nTest score: ", evaluation[0])
print("Test accuracy: \n", evaluation[1])

    ###### PREDICTIONS ######
predictions = my_model.predict(X_test)

vrai_positifs = 0
faux_positifs = 0
vrai_negatifs = 0
faux_negatifs = 0

for i in range(predictions.shape[0]):
    maxi = max(predictions[i,:])
    if maxi == predictions[i, 0]:
        classe = 0
    elif maxi == predictions[i,1]:
        classe = 1
    elif maxi == predictions[i,2]:
        classe = 2
        
    if (encoded_Y_test[i, 0] == 1.0) and (classe == 0):
        vrai_positifs += 1
    elif (encoded_Y_test[i, 1] == 1.0) and (classe == 1):
        vrai_positifs += 1
    elif (encoded_Y_test[i, 2] == 1.0) and (classe == 0):
        faux_positifs += 1
    elif (encoded_Y_test[i, 2] == 1.0) and (classe == 1):
        faux_positifs += 1
    elif (encoded_Y_test[i, 2] == 1.0) and (classe == 2):
        vrai_negatifs += 1
    elif (encoded_Y_test[i, 2] == 0.0) and (classe == 2):
        faux_negatifs += 1

print("\nPourcentage de vrais positifs: {:.2f}%".format(vrai_positifs*100/len(predictions)))
print("Pourcentage de faux positifs: {:.2f}%".format(faux_positifs*100/len(predictions)))
print("Pourcentage de vrais negatifs: {:.2f}%".format(vrai_negatifs*100/len(predictions)))
print("Pourcentage de faux negatifs: {:.2f}%\n".format(faux_negatifs*100/len(predictions)))

    ### NUCLEOTIDES
y_test_prediction_nucleotide = predictions[:,0]
y_test_nucleotide = encoded_Y_test[:,0]
nucleotide_fpr, nucleotide_tpr, nucleotide_thresholds = metrics.roc_curve(y_test_nucleotide, y_test_prediction_nucleotide, pos_label=2)
nucleotide_roc_auc = metrics.auc(nucleotide_fpr, nucleotide_tpr)

    ### HEME
y_test_prediction_heme = predictions[:,1]
y_test_heme = encoded_Y_test[:,1]
heme_fpr, heme_tpr, heme_thresholds = metrics.roc_curve(y_test_heme, y_test_prediction_heme, pos_label=2)
heme_roc_auc = metrics.auc(heme_fpr, heme_tpr)

    ### CONTROL
y_test_prediction_control = predictions[:,2]
y_test_control = encoded_Y_test[:,2]
control_fpr, control_tpr, control_thresholds = metrics.roc_curve(y_test_control, y_test_prediction_control, pos_label=2)
control_roc_auc = metrics.auc(control_fpr, control_tpr)

    ### Courbes de ROC: evaluate classifier output quality using cross-validation
plt.title("Courbe ROC")

plt.plot(control_fpr,
        control_tpr,
        color='blue',
        linestyle='-',
        linewidth=2,
        label = "control AUC = %0.2f" % control_roc_auc)

plt.plot(heme_fpr,
        heme_tpr,
        color='purple',
        linestyle='-',
        linewidth=2,
        label = "heme AUC = %0.2f" % heme_roc_auc)

plt.plot(nucleotide_fpr,
        nucleotide_tpr,
        color='orange',
        linestyle='-',
        linewidth=2,
        label = "nucleotide AUC = %0.2f" % nucleotide_roc_auc)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.ylabel("Sensibility")
plt.xlabel("1-Sensibility")

plt.show()