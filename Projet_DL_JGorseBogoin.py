#Importation des modules
import numpy as np 
import os
import random
import keras

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution3D, MaxPooling3D, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K



def creation_control (nbre_individu):
    #Génération des listes contenant 50 poches choisies au hasard 
    echantillon_control = random.sample(control_list, nbre_individu)
    random.shuffle(echantillon_control)
    return echantillon_control

def creation_heme (nbre_individu):
    echantillon_heme = random.sample(heme_list, nbre_individu)
    random.shuffle(echantillon_heme)
    return echantillon_heme

def creation_steroid (nbre_individu):
    echantillon_steroid = random.sample(steroid_list, nbre_individu)
    random.shuffle(echantillon_steroid)
    return echantillon_steroid

def creation_nucleotide (nbre_individu): 
    echantillon_nucleotide = random.sample(nucleotide_list, nbre_individu)  
    random.shuffle(echantillon_nucleotide)
    return echantillon_nucleotide

def generation_voxels (voxels_tot, steroid, heme, nucleotide, control):
    #Génération de la liste des voxels
    echantillon_voxel = []
    for voxel in voxels_tot:
        if voxel.replace('.npy','') in steroid:
            echantillon_voxel.append(voxel)
        if voxel.replace('.npy','') in heme:
            echantillon_voxel.append(voxel)
        if voxel.replace('.npy','') in nucleotide:
            echantillon_voxel.append(voxel)
        if voxel.replace('.npy','') in control:
            echantillon_voxel.append(voxel)
    random.shuffle(echantillon_voxel)
    return echantillon_voxel

def definition_de_x(path, echantillon_voxel):
    try:
        X = [np.load("{}/{}".format(path, voxel))
             for voxel in echantillon_voxel]
    except ValueError:
        print("{}".format(voxel))
    X = [np.squeeze(array) for array in X]
    X = np.array(X)
    X = np.moveaxis(X, 1, -1)
    return X

def definition_de_y(echantillon_voxel, nucleotide, heme, control, steroid):
    Y = []
    for voxel in echantillon_voxel:
        if voxel.replace('.npy','') in nucleotide:
            Y.append(1)
        elif voxel.replace('.npy','') in heme:
            Y.append(2)
        elif voxel.replace('.npy','') in steroid:
            Y.append(4)
        elif voxel.replace('.npy','') in control:
            Y.append(3)
    Y  = np.array(Y)
    return Y

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
        model.add(Dense(128)) # TODO changed to 64 for the CAM
        model.add(LeakyReLU(alpha = 0.1))
        # Dropout 3
        model.add(Dropout(0.4))
        # Fully connected layer 2 to shape (2) for 2 classes
        model.add(Dense(2))
        model.add(Activation('softmax'))
        return model   


###### MAIN ######

#Génération des listes poches
control_list = []
heme_list = []
steroid_list = []
nucleotide_list = []

with open ("/Users/julieb/Data_DL/control_list.txt", "r") as fillin:
    for ligne in fillin:
        control_list.append(ligne.replace('\n',''))
        
with open ("/Users/julieb/Data_DL/heme_list.txt", "r") as fillin:
    for ligne in fillin:
        heme_list.append(ligne.replace('\n',''))

with open ("/Users/julieb/Data_DL/steroid_list.txt", "r") as fillin:
    for ligne in fillin:
        steroid_list.append(ligne.replace('\n',''))

with open ("/Users/julieb/Data_DL/nucleotide_list.txt", "r") as fillin:
    for ligne in fillin:
        nucleotide_list.append(ligne.replace('\n',''))

#Génération de la liste Voxels
voxels_path = "/Users/julieb/Data_DL/Voxels"
voxels_tot = os.listdir(voxels_path)
voxels_tot = list(filter(lambda fichier: fichier.endswith('.npy'), voxels_tot))


###### JEU D'APPRENTISSAGE ######
train_control = creation_control(40)
train_heme = creation_heme(40)
train_steroid = creation_steroid(40)
train_nucleotide = creation_nucleotide(40)
train_voxels = generation_voxels(voxels_tot, train_steroid, train_heme, train_nucleotide, train_control)

#Chargement des voxels
X_train = definition_de_x(voxels_path, train_voxels)
Y_train = definition_de_y(train_voxels, train_nucleotide, train_heme, train_control, train_steroid)
encoded_Y_train = to_categorical(Y_train)

###### JEU DE TEST ######
test_control = creation_control(69)
test_heme = creation_heme(69)
test_steroid = creation_steroid(69)
test_nucleotide = creation_nucleotide(69)
test_voxels = generation_voxels(voxels_tot, test_steroid, test_heme, test_nucleotide, test_control)

#Chargement des voxels
X_test = definition_de_x(voxels_path, test_voxels)
Y_test = definition_de_y(test_voxels, test_nucleotide, test_heme, test_control, test_steroid)
encoded_Y_test = to_categorical(Y_test)






