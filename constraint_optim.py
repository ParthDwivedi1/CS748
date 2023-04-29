import numpy as np
import json 
import logging
import matplotlib.pyplot as plt


#formulating problem as constraint minimization problem::
def get_l(state,alpha,prob,cost):
    loss=0
    mtx=np.array([1-state, state])
    for i in range(alpha-1):
        mtx=np.matmul(mtx,prob)
        loss+= np.min(mtx)
    mtx=np.matmul(mtx,prob)
    return loss+cost,mtx
def optim_con(prob,cost=0.4,Mx_val=10):
    #Mx_val=50
    arr=np.zeros((Mx_val,Mx_val))
    for i in range(1,Mx_val+1):
        for j in range(1,Mx_val+1):
            alpha_l,P_0=get_l(0,i,prob,cost)
            beta_l,P_1=get_l(1,j,prob,cost)
            v=(alpha_l*P_1[0]+P_0[1]*beta_l)/(i*P_1[0]+j*P_0[1])
            arr[i-1,j-1]=v
            #print(v,end=" ")
        #print("")
    res=np.unravel_index(np.argmin(arr),arr.shape)
    #print(arr)
    alpha=res[0]+1
    beta=res[1]+1
    return alpha,beta,np.min(arr)
    #print(f"alpha:{alpha} \nbeta:{beta}")

def get_min_l(test_case,state):
    P=test_case['prob']
    cost=test_case['cost']
    T=test_case['T']
    mtx=np.zeros((1,2))
    mtx[0,state]=1
    loss=0
    vals=np.zeros(T)
    for i in range(T):
        loss+=np.min(mtx)
        vals[i]=(loss+cost)/(i+1)
        mtx=np.matmul(mtx,P)
    return np.argmin(vals)+1


#predicted algorithm (similar to back-calculate::)
# def get_new_algo(P,state,cost,T):
#     mtx=np.zeros((1,2))
#     mtx[0,state]=1
#     loss=0
#     for i in range(T):
#         loss+=np.min(mtx)
#         if((loss+cost)/(i+1)<(loss))
