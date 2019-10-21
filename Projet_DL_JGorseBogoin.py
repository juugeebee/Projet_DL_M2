#Importation des modules
import numpy as np 
import os
import random
import keras

from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense, Flatten 
from keras.layers import add, Activation
from keras.layers import Conv1D, AveragePooling1D
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

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

def definition_de_x(path, chosen_pocket):
    try:
        X = [np.load("{}/{}".format(path, pocket))
             for pocket in chosen_pocket]
    except ValueError:
        print("{}".format(pocket))
    X = [np.squeeze(array) for array in X]
    X = np.array(X)
    X = np.moveaxis(X, 1, -1)
    return X

def definition_de_y(chosen_pocket, nucleotid, heme, control, steroid):
    Y = []
    for pocket in chosen_pocket:
        if pocket in nucleotid:
            Y.append(1)
        elif pocket in heme:
            Y.append(2)
        elif pocket in steroid:
            Y.append(4)
        elif pocket in control:
            Y.append(3)
    Y  = np.array(Y)
    return Y

###### MAIN ######

#Génération des listes Poches
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

train_control = creation_control(50)
train_heme = creation_heme(50)
train_steroid = creation_steroid(50)
train_nucleotide = creation_nucleotide(50)
train_voxels = generation_voxels(voxels_tot, train_steroid, train_heme, train_nucleotide, train_control)

#Chargement des voxels
X_train = definition_de_x(voxels_path, train_voxels)
Y_train = definition_de_y(train_voxels, nucleotide_list, heme_list, control_list, steroid_list)






