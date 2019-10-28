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
        filters=14,
        kernel_size=5,
        padding='same',   
        data_format='channels_first',
        activation = 'relu'
    ))
    # Dropout 1
    model.add(Dropout(0.2))
    # Maxpooling 1
    model.add(MaxPooling3D(
        pool_size=(2,2,2),
        strides=None,
        padding='same',    # Padding method
        data_format='channels_first'
    ))
    # Dropout 2
    model.add(Dropout(0.4))
    # FC 1
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu')) 
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

        #Génération d'un jeu de données avec 200 poches
jeu_control = creation_control(200)
jeu_heme = creation_heme(200)
jeu_nucleotide = creation_nucleotide(200)

    ###### JEU D'APPRENTISSAGE ######
train_control = []
train_heme = []
train_nucleotide = []

train_control = jeu_control[:51]
train_heme = jeu_heme[:51]
train_nucleotide = jeu_nucleotide[:51]

train_voxels = generation_voxels(voxels_tot, train_heme, train_nucleotide, train_control)

X_train = definition_de_x(voxels_path, train_voxels)
print("\nFormat du X_train: {}".format(X_train.shape))
Y_train = definition_de_y(train_voxels, train_nucleotide, train_heme, train_control)
encoded_Y_train = to_categorical(Y_train)
print("Format du Y_train: {}\n".format(encoded_Y_train.shape))

    ###### JEU DE TEST ######
test_control = []
test_heme = []
test_nucleotide = []

test_control = jeu_control[50:201]
test_heme = jeu_heme[50:201]
test_nucleotide = jeu_heme[50:201]

test_voxels = generation_voxels(voxels_tot, test_heme, test_nucleotide, test_control)

X_test = definition_de_x(voxels_path, test_voxels)
Y_test = definition_de_y(test_voxels, test_nucleotide, test_heme, test_control)
encoded_Y_test = to_categorical(Y_test)

    ###### CREATION DU MODEL ######
my_model = creation_modele()
print(my_model.summary())
my_model.compile(optimizer=keras.optimizers.Adam(lr=0.00001),loss="categorical_crossentropy",metrics=['accuracy'])


history = my_model.fit(X_train, encoded_Y_train, epochs = 10, batch_size = 20,
             validation_split = 0.1)

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


# Plot all ROC curves
plt.figure()
plt.plot(nucleotide_fpr, nucleotide_tpr,
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(nucleotide_roc_auc),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(heme_fpr, heme_tpr,
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(heme_roc_auc),
         color='navy', linestyle=':', linewidth=4)

plt.plot(control_fpr, control_tpr,
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(control_roc_auc),
         color='navy', linestyle=':', linewidth=4)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Courbes de ROC')
plt.legend(loc="lower right")
plt.show()
