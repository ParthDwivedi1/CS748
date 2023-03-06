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

# def DP(mdp_history,prob,cost,k_max=100):
#     horizon=np.size(mdp_history)
#     arr=np.zeros((horizon,k_max+1,horizon))+np.inf# DP array i,j,k represents min cost of [0-i] with j vals and last probe at k::

#     arr[0,1,0]=0
#     for j in range(horizon):
#         print(f"{j}/{horizon}")
#         for k in range(horizon-1,-1,-1):
#             # for j in range(min(k+2,k_max+1)):
#             #     if(j==0):
#             #         arr[i,j,k]=query_loss(0,i,mdp_history[0],prob)
#             #     elif(k==i):
#             #         if(i==0):
#             #             arr[i,j,k]=min(arr[i,j,k],0+cost)
#             #         else:
#             #             arr[i,j,k]=min(arr[i,j,k],np.min(arr[i-1,j-1,max(j-2,0):i])+cost)
#             #     elif(k<i):
#             #         arr[i,j,k]=min(arr[i,j,k],arr[k,j,k]+query_loss(k,i,mdp_history[k],prob))
#             #k_max=int(np.min(arr[arr>=0])//cost)#updating k_max at the end of each iteration::# can be improvised
#             for i in range(horizon-1,k,-1):
#                 arr[i,j,k]=min(arr[i,j,k],arr[k,j,k]+query_loss(k,i,mdp_history[k],prob));
#                 if(j!=horizon-1):
#                     arr[i,j+1,i]=min(arr[i,j+1,i],arr[i-1,j,k]+cost);
#     print(arr)
#     best_loss=(np.min(arr[horizon-1,:,:]))
#     print(best_loss)
#     return best_loss

#DP(np.zeros((100)),np.array([[0.8,0.2],[0.7,0.3]]),5)

def loss_cal(prob,i,j,state): #i is position of last query, j is last state at which we need loss
    
    n=prob.shape[1]
    loss = np.zeros((n,1));
    # print(loss.shape)
    for x in range(j-i):
        temp = np.reshape(np.min(np.linalg.matrix_power(prob,x+1),axis=1),(2,1))
        # print(temp.shape)
        loss+= temp
    return np.matmul(state,loss)

def OPTPolicy(prob,T,cost):
    n = prob.shape[0]
    
    dp = np.zeros((T+1,n)) # dp_ik is the position to query next if previous query was at i and result was state k   
    loss = np.zeros((T+1,n)) # Expected loss till the end of the horizon
    for i in range(n):
        dp[T][i]=np.inf
        loss[T][i]=0

    for i in range(T-1,-1,-1):
        for k in range(n):
            state = np.zeros((1,n))
            state[0,k]=1
            curr=loss_cal(prob,i,T,state)
            dp[i,k]=np.inf
            for x in range(i+1,T+1):
                new_curr = loss_cal(prob,i,x-1,state) + cost
                
                state = np.matmul(state,np.linalg.matrix_power(prob,x-i))
                for s in range(n):
                    new_curr += loss[x,s]*state[0,s]

                if new_curr < curr:
                    curr=new_curr
                    dp[i,k]=x
                    
            loss[i,k]=curr
            
    print("Expected Loss: ",loss[0,0])
    # print(dp)
    # curr=0
    # while(curr<1000):
    #     print(dp[int(curr),1])
    #     curr=dp[int(curr),1]