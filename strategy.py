import numpy as np

def query_loss(i,j,res,prob):
        #this is a quey to return loss from i-j inclusive if a query is made given outcome
        state_curr=[1-res,res]
        loss=0
        for idx in range(i+1,j+1):
            temp = np.matmul(state_curr, prob)
            loss+=np.min(temp)
        return loss
def DP(mdp_history,prob,cost,k_max=100):
    horizon=np.size(mdp_history)
    arr=np.zeros((horizon,k_max+1,horizon))-1# DP array i,j,k represents min cost of [0-i] with j vals and last probe at k::

    for i in range(horizon):
        print(f"{i}/{horizon}")
        for k in range(i+1):
            for j in range(min(k+2,k_max+1)):
                if(j==0):
                    arr[i,j,k]=query_loss(0,i,mdp_history[0],prob)
                elif(k==i):
                    if(i==0):
                        arr[i,j,k]=0+cost
                    else:
                        arr[i,j,k]=np.min(arr[i-1,j-1,max(j-2,0):i])+cost
                elif(k<i):
                    arr[i,j,k]=arr[k,j,k]+query_loss(k,i,mdp_history[k],prob)
            #k_max=int(np.min(arr[arr>=0])//cost)#updating k_max at the end of each iteration::# can be improvised
    best_loss=(np.min(arr[horizon-1,arr[horizon-1,:,:]>=0]))
    print(best_loss)
    return best_loss

#DP(np.zeros((100)),np.array([[0.8,0.2],[0.7,0.3]]),5)





