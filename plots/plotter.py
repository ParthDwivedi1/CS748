import numpy as np
import json 
import logging
import matplotlib.pyplot as plt
import sys

sys.path.insert(0,'../')
from utility import *


BASE_DIR='../'

def plot_g1(fn):
    arr=read_ndjson(BASE_DIR+fn)
    plot_y_inf=[]
    plot_y_fin=[]
    plot_x=[]
    for t in arr:
        #print(t)
        plot_x.append(t['tc']['T'])
        plot_y_inf.append(t['loss_infinite'])
        plot_y_fin.append(t['loss_finite'])
    plt.plot(plot_x,plot_y_inf,'r-',plot_x,plot_y_fin,'b-')
    plt.legend(["constraint_optim","DP"])
    plt.show()


def unsym_index(A):
    A=np.array(A)
    norm_sym=np.linalg.norm(A+A.T,2)
    norm_anti=np.linalg.norm(A-A.T,2)
    return (norm_sym-norm_anti)/(norm_sym+norm_anti)
def plot_table():
    arr=read_ndjson('../test_algosymm_1.json')
    print(f"symmetry index    ALGO    HAPI(exact)")
    for obj in arr:
        print(f"{round(unsym_index(obj['test_case']['prob']),2):10} {round(obj['loss_symmalgo'][0],2):10} {round(obj['loss_inf'],2):10}")




if __name__=='__main__':
    plot_table()


