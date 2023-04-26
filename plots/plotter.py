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









if __name__=='__main__':
    plot_g1('test_dpvsinf.json')


