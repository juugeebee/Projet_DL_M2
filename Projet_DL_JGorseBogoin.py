#Importation des modules
import numpy as np
from numpy import loadtxt
import pandas 
import keras
import os
import random
from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense, Flatten 
from keras.layers import add, Activation
from keras.layers import Conv1D, AveragePooling1D
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

#Génération des listes Poches
control_list = []
heme_list = []
steroid_list = []
nucleotide_list = []

with open ("/Users/julieb/M2-BIB/DL_2.0/Projet_DL/Data/control_list.txt", "r") as fillin:
    for ligne in fillin:
        control_list.append(ligne.replace('\n',''))
        
with open ("/Users/julieb/M2-BIB/DL_2.0/Projet_DL/Data/heme_list.txt", "r") as fillin:
    for ligne in fillin:
        heme_list.append(ligne.replace('\n',''))

with open ("/Users/julieb/M2-BIB/DL_2.0/Projet_DL/Data/steroid_list.txt", "r") as fillin:
    for ligne in fillin:
        steroid_list.append(ligne.replace('\n',''))

with open ("/Users/julieb/M2-BIB/DL_2.0/Projet_DL/Data/nucleotide_list.txt", "r") as fillin:
    for ligne in fillin:
        nucleotide_list.append(ligne.replace('\n',''))

#Génération de la liste Voxels
voxels_path = "/Users/julieb/M2-BIB/DL_2.0/Projet_DL/Data/Voxels"
voxels_tot = os.listdir(voxels_path)
voxels_tot = list(filter(lambda fichier: fichier.endswith('.npy'), voxels_tot))

###### JEU D'APPRENTISSAGE ######

#Génération des listes contenant 50 poches choisies au hasard 
echantillon_control = random.sample(control_list, 50)
random.shuffle(echantillon_control)

echantillon_heme = random.sample(heme_list, 50)
random.shuffle(echantillon_heme)

echantillon_steroid = random.sample(steroid_list, 50)
random.shuffle(echantillon_steroid)

echantillon_nucleotide = random.sample(nucleotide_list, 50)  
random.shuffle(echantillon_nucleotide)

#Génération de la liste des voxels
echantillon_voxel = []
for voxel in voxels_tot:
    if voxel.replace('.npy','') in echantillon_steroid:
        echantillon_voxel.append(voxel)
    if voxel.replace('.npy','') in echantillon_heme:
        echantillon_voxel.append(voxel)
    if voxel.replace('.npy','') in echantillon_nucleotide:
        echantillon_voxel.append(voxel)
    if voxel.replace('.npy','') in echantillon_control:
        echantillon_voxel.append(voxel)
random.shuffle(echantillon_voxel)






