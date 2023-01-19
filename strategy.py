import numpy as np

def query_loss(i,j,res,prob):
        #this is a quey to return loss from i-j inclusive if a query is made given outcome
        state_curr=np.array([[1-res],[res]])
        loss=0
        for idx in range(i+1,j+1):
            state_curr = np.matmul(prob,state_curr)
            loss+=np.min(state_curr)
        # print("I: ",i,"J: ",j,"LOSS: ",loss)
        return loss
# def DP(mdp_history,prob,cost,k_max=100):
#     horizon=np.size(mdp_history)
#     arr=np.zeros((horizon,k_max+1,horizon))+np.inf# DP array i,j,k represents min cost of [0-i] with j vals and last probe at k::

#     for i in range(horizon):
#         print(f"{i}/{horizon}")
#         for k in range(i+1):
#             for j in range(min(k+2,k_max+1)):
#                 if(j==0):
#                     arr[i,j,k]=query_loss(0,i,mdp_history[0],prob)
#                 elif(k==i):
#                     if(i==0):
#                         arr[i,j,k]=min(arr[i,j,k],0+cost)
#                     else:
#                         arr[i,j,k]=min(arr[i,j,k],np.min(arr[i-1,j-1,max(j-2,0):i])+cost)
#                 elif(k<i):
#                     arr[i,j,k]=min(arr[i,j,k],arr[k,j,k]+query_loss(k,i,mdp_history[k],prob))
#             #k_max=int(np.min(arr[arr>=0])//cost)#updating k_max at the end of each iteration::# can be improvised
#     best_loss=(np.min(arr[horizon-1,:,:]))
#     print(best_loss)
#     return best_loss

def DP(mdp_history,prob,cost,k_max=100):
    horizon=np.size(mdp_history)
    arr=np.zeros((horizon,k_max+1,horizon))+np.inf# DP array i,j,k represents min cost of [0-i] with j vals and last probe at k::

    arr[0,1,0]=0
    for j in range(horizon):
        print(f"{j}/{horizon}")
        for k in range(horizon-1,-1,-1):
            # for j in range(min(k+2,k_max+1)):
            #     if(j==0):
            #         arr[i,j,k]=query_loss(0,i,mdp_history[0],prob)
            #     elif(k==i):
            #         if(i==0):
            #             arr[i,j,k]=min(arr[i,j,k],0+cost)
            #         else:
            #             arr[i,j,k]=min(arr[i,j,k],np.min(arr[i-1,j-1,max(j-2,0):i])+cost)
            #     elif(k<i):
            #         arr[i,j,k]=min(arr[i,j,k],arr[k,j,k]+query_loss(k,i,mdp_history[k],prob))
            #k_max=int(np.min(arr[arr>=0])//cost)#updating k_max at the end of each iteration::# can be improvised
            for i in range(horizon-1,k,-1):
                arr[i,j,k]=min(arr[i,j,k],arr[k,j,k]+query_loss(k,i,mdp_history[k],prob));
                if(j!=horizon-1):
                    arr[i,j+1,i]=min(arr[i,j+1,i],arr[i,j,k]+cost);
    # print(arr)
    best_loss=(np.min(arr[horizon-1,:,:]))
    print(best_loss)
    return best_loss

#DP(np.zeros((100)),np.array([[0.8,0.2],[0.7,0.3]]),5)





