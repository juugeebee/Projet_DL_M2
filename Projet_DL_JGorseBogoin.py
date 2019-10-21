#Importation des modules
import numpy as np
from numpy import loadtxt
import pandas 
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
import os

#Génération des listes Poches
control_list = []
heme_list = []
steroid_list = []
nucleotid_list = []

with open ("/Users/julieb/M2-BIB/DL_2.0/Projet_DL/Data/control_list.txt", "r") as fillin:
    for ligne in fillin:
        control_list.append(ligne)

with open ("/Users/julieb/M2-BIB/DL_2.0/Projet_DL/Data/heme_list.txt", "r") as fillin:
    for ligne in fillin:
        heme_list.append(ligne)

with open ("/Users/julieb/M2-BIB/DL_2.0/Projet_DL/Data/steroid_list.txt", "r") as fillin:
    for ligne in fillin:
        steroid_list.append(ligne)

with open ("/Users/julieb/M2-BIB/DL_2.0/Projet_DL/Data/nucleotide_list.txt", "r") as fillin:
    for ligne in fillin:
        nucleotid_list.append(ligne)

#Génération de la liste Voxels
voxels_path = "/Users/julieb/M2-BIB/DL_2.0/Projet_DL/Data/Voxels"
voxels_tot = os.listdir(voxels_path)
voxels_tot = list(filter(lambda fichier: fichier.endswith('.npy'), voxels_tot))





