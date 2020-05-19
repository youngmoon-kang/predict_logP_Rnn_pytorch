# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:41:10 2020

@author: SFC202004009
"""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

atoms = ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
        'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
        'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
        'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']

def read_smiles(file_path, number):
    f = open(file_path, 'r')
    smiles = f.readlines()
    
    smi_list = []
    for i, smi in enumerate(smiles):
        if (i == number):
            break
        if (smi.find('I') == -1):
            s = smi.split()[0]
            s = Chem.MolToSmiles(Chem.MolFromSmiles(s))
            smi_list.append(s)

    return smi_list
        
def get_logP(smi_list):
    logP_list = []
    
    for smi in smi_list:
        m = Chem.MolFromSmiles(smi)
        logP_list.append(MolLogP(m))
    
    return logP_list

def get_fingerprint(smi_list):
    fingerprints = []
    
    for smi in smi_list:
        m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m,2)
        arr = np.array((1, ))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprints.append(fp)

    fingerprints = np.asarray(fingerprints).astype(float)
    return fingerprints

def get_graph(smi_list):
    max_atoms = 50
    adj_list = []
    feature_list = []
    
    for smi in smi_list:
        m = Chem.MolFromSmiles(smi)
        adj = Chem.rdmolops.GetAdjacencyMatrix(m)
        
        if(adj.shape[0] <= max_atoms):
            feature= np.zeros((max_atoms, 58))
            temp = []
            for a in m.GetAtoms():
                temp.append( atom_feature(a) )
            feature[0:len(temp), 0:58] = temp
            feature_list.append(feature)
            
            adj[0:adj.shape[0], 0:adj.shape[0]] = adj + np.eye(adj.shape[0])
            adj_list.append(adj)
    
    return feature_list, adj_list
        
def atom_feature(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                                       'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                                       'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                       'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])    # (40, 6, 5, 6, 1)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    #print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
    
def draw_2D(smi_list):
    for smi in smi_list:
        m = Chem.MolFromSmiles(smi)
        img = Draw.MolToImage(m)
        plt.title(smi)
        plt.imshow(img)
        
def smiles_to_number(smi_list, max_len = 120):
    max_number = max_len
    vocab = [' ', 'c', 'C', '(', ')', 'O' ,'1' ,'2', 'N', 
             '=', '[', ']', '@' ,'3' ,'H',
             'n', '4', 'F', '+', 'S', 'l', 's' ,'/', 'o' ,
             '-' ,'5' ,'#', 'B' ,'r' ,'\\', '6']
    result = []
    for smi in smi_list:
        if(smi.find('I') != -1):
            continue
        smi_max = ' ' * (max_number - len(smi))
        smi_max = smi + smi_max
        temp = []

        for i in smi_max:
            temp.append(vocab.index(i))
        result.append(temp)
        
    return result
            
def make_one_hot(smi, num):
    one = np.identity(num, dtype = np.float)
    nump_one_hot = np.zeros((smi.shape[0], smi.shape[1], num), dtype = np.float)
    for i_n, i in enumerate(smi):
        for j_n, j in enumerate(i):
            nump_one_hot[i_n, j_n] = one[j]
    return nump_one_hot
    
if (__name__ == '__main__'):
    smiles = read_smiles('ZINC.smiles',500)
    result = smiles_to_number(smiles)
    print(smiles[0])
    print(result[0])
    onehot = make_one_hot(np.array(result), 31)
    print(onehot.shape)