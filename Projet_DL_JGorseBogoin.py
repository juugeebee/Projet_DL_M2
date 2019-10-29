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

from sklearn.preprocessing import LabelEncoder

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

    ###### CREATION DU JEU DE DONNEES X ######
def definition_de_x(path, echantillon_voxel):
    X = [np.load("{}/{}".format(path, voxel)) for voxel in echantillon_voxel]
    X = [np.squeeze(array) for array in X]
    X = np.array(X)
    return X

    ###### CREATION DU JEU DE DONNEES Y ######
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

    ###### CREATION DU MODELE ######
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

def table_de_confusion (encoded_Y_test, predictions, classe):
    vrai_positifs = 0
    faux_positifs = 0
    vrai_negatifs = 0
    faux_negatifs = 0
    sensibilite = []
    specificite = []

    for threshold in (np.arange(0,1,0.001)):
        for i in range(len(predictions)):
            if predictions[i,classe] > threshold:
                if encoded_Y_test[i,classe] == 1.0 and predictions[i,classe] == max(predictions[i,:]):
                    vrai_positifs = vrai_positifs + 1
                else:
                    faux_positifs = faux_positifs + 1
            else:
                if encoded_Y_test[i,classe] == 0.0 and predictions[i,classe] != max(predictions[i,:]):
                    vrai_negatifs = vrai_negatifs + 1
                else:
                    faux_negatifs = faux_negatifs + 1
        
        sensibilite.append(vrai_positifs/(vrai_positifs + faux_negatifs))
        specificite.append(1-(vrai_negatifs/(vrai_negatifs + faux_positifs))) 
    
    return specificite, sensibilite  

def calcul_auc (specificite, sensibilite): 
    auc = 0
    for i in range(len(sensibilite)-1):
        rectangle = sensibilite[i] * (specificite[i] - specificite[i+1])
        auc = auc + rectangle
    return auc 


###### MAIN ######

        #Génération des listes poches
control_list = []
heme_list = []
nucleotide_list = []

with open ("./Data/control.list", "r") as fillin:
    for ligne in fillin:
        control = ligne.replace('\n','')
        control_list.append(control)
        
with open ("./Data/heme.list", "r") as fillin:
    for ligne in fillin:
        heme = ligne.replace('\n','')
        heme_list.append(heme)

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
classes = LabelEncoder()
integer_encoding = classes.fit_transform(Y_train)
encoded_Y_train = to_categorical(integer_encoding)
print("Format du Y_train: {}\n".format(encoded_Y_train.shape))


    ###### JEU DE TEST ######
test_control = []
test_heme = []
test_nucleotide = []

test_control = jeu_control[50:201]
test_heme = jeu_heme[50:201]
test_nucleotide = jeu_nucleotide[50:201]

test_voxels = generation_voxels(voxels_tot, test_heme, test_nucleotide, test_control)

X_test = definition_de_x(voxels_path, test_voxels)

Y_test = definition_de_y(test_voxels, test_nucleotide, test_heme, test_control)
# Colonne 0: NUCLEOTIDE
# Colonne 1: HEME
# Colonne 2: CONTROL
classes = LabelEncoder()
integer_encoding = classes.fit_transform(Y_test)
encoded_Y_test = to_categorical(Y_test)


    ###### CREATION DU MODELE ######
my_model = creation_modele()
print(my_model.summary())

my_model.compile(optimizer=keras.optimizers.Adam(lr=0.00001),
                loss="categorical_crossentropy",
                metrics=['accuracy'])

history = my_model.fit(X_train, encoded_Y_train, epochs = 10, batch_size = 20,
             validation_split = 0.1)

        #Sauvegarder le model
my_model.save('./my_model.h5')

    ###### EVALUATION DU MODEL ######
evaluation = my_model.evaluate(X_test, encoded_Y_test)
print("\nTest score: {:.2f}".format(evaluation[0]))
print("Test accuracy: {:.2f}".format(evaluation[1]))


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

vp = vrai_positifs*100/len(predictions)
fp = faux_positifs*100/len(predictions)
vn = vrai_negatifs*100/len(predictions)
fn = faux_negatifs*100/len(predictions)

print("\nPourcentage de vrais positifs: {:.2f}%".format(vp))
print("Pourcentage de faux positifs: {:.2f}%".format(fp))
print("Pourcentage de vrais negatifs: {:.2f}%".format(vn))
print("Pourcentage de faux negatifs: {:.2f}%\n".format(fn))


    ### NUCLEOTIDES
nucleotide_sensibilite, nucleotide_specificite = table_de_confusion(encoded_Y_test, predictions, 0)

nucleotide_auc = calcul_auc(nucleotide_specificite, nucleotide_sensibilite)
print("Aire sous la courbe classe 'Nucleotides': {:.2f}".format(nucleotide_auc))


     ### HEME
heme_sensibilite, heme_specificite = table_de_confusion(encoded_Y_test, predictions, 1)

heme_auc = calcul_auc(heme_specificite, heme_sensibilite)
print("Aire sous la courbe classe 'Hemes': {:.2f}\n".format(heme_auc))

    ###Representation des courbes de ROC
plt.title('Courbes de ROC')
plt.plot(heme_specificite, heme_sensibilite, label='Hemes', color='blue', linestyle='-', linewidth=2)
plt.plot(nucleotide_specificite, nucleotide_sensibilite, label='Nucleotides', color='purple', linestyle='-', linewidth=2)
plt.legend(loc="lower right")
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('1- Specificite')
plt.ylabel('Sensibilite')
plt.savefig("./ROC")
plt.show()