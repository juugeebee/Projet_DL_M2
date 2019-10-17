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

#Acces aux fichiers et generation des listes
path = os.getcwd()
oknotok = os.access(path + "/Data", mode=os.R_OK)
if oknotok == FALSE:
    print("Le repertoire Data n'existe pas. Le programme va s'arreter!")
    break

data_path = os.chroot(path + "/Data")
print(data_path)

control_file = "control.list.txt"
control_list = []
heme_file = "heme.list.txt"
heme_list = []
steroid_file = "steroid.list.txt"
steroid_list = []
nucleotid_file = "nucleotid.list.txt"
nucleotid_list = []

with open (control_file, "r") as fillin:
    for ligne in fillin:
        control_list.append(ligne)

with open (heme_file, "r") as fillin:
    for ligne in fillin:
        heme_list.append(ligne)

with open (steroid_file, "r") as fillin:
    for ligne in fillin:
        steroid_list.append(ligne)

with open (nucleotid_file, "r") as fillin:
    for ligne in fillin:
        nucleotid_list.append(ligne)

oknotok = os.access(data_path + "/voxels", mode=os.R_OK)
if oknotok == FALSE:
    print("Le repertoire voxels n'existe pas. Le programme va s'arreter!")
    break

voxel_path = os.chroot(data_path + "/voxels")
voxel_list = os.listdir(voxel_path)



