import numpy as np

def HAPI(test_case):
    prob = test_case['prob']
    
    horizon =test_case['T']
    cost = test_case['cost']
        
    old_policy=np.array([horizon,horizon])
    
    pow_prob = np.zeros((1001,2,2))
    
    pow_prob[0]=prob
    
    for i in range(1,horizon+1):
        pow_prob[i,:,:]=np.matmul(pow_prob[i-1,:,:],prob)
        # print(pow_prob[i,:,:])
        

    new_policy=np.array([0,0]) # Sample after itne steps
    # print(pow_prob[6,:,:])
    while((new_policy!=old_policy).any()):
        loss=0
        loss_al = np.sum(np.min(pow_prob[:new_policy[0]],axis=1),axis=0)
        loss_bet = np.sum(np.min(pow_prob[:new_policy[1]],axis=1),axis=0)
        
        # print("NUM",((loss[0]+cost)/(new_policy[0]+1) - (loss[1]+cost)/(new_policy[1]+1)),"DEN",(pow_prob[new_policy[0]+1,1,0]+pow_prob[new_policy[1]+1,0,1]))
        orig = ((loss_al[0]+cost)/(new_policy[0]+1) - (loss_bet[1]+cost)/(new_policy[1]+1))/(pow_prob[new_policy[0]+1,1,0]+pow_prob[new_policy[1]+1,0,1])
        
        c = (pow_prob[new_policy[1]+1,0,1]*((loss_al[0]+cost)/(new_policy[0]+1)) + pow_prob[new_policy[0]+1,1,0]*((loss_bet[1]+cost)/(new_policy[1]+1)))/(pow_prob[new_policy[0]+1,1,0]+pow_prob[new_policy[1]+1,0,1])
        
        #print(c)
        
        newal=-1
        newb=-1
        
        # print(orig)
        
        for alpha in range(horizon):
            c_los= np.sum(np.min(pow_prob[:alpha],axis=1),axis=0)
            
            # print((c_los[0]+cost)/(alpha+1)+pow_prob[alpha+1,0,0]*orig,orig+c)
            
            if(newal==-1 and (c_los[0]+cost)/(alpha+1)+pow_prob[alpha+1,0,0]*orig<orig+c):
                newal=alpha
            
            # print((c_los[1]+cost)/(alpha+1)+pow_prob[alpha+1,0,1]*orig,c)
            
            if(newb==-1 and (c_los[1]+cost)/(alpha+1)+pow_prob[alpha+1,0,1]*orig<c):
                newb=alpha

            if(newal!=-1 and newb !=-1):
                break
            
        # print(newal,newb)
        if(newal==-1 and newb==-1):
            break;
        else:
            old_policy[0]=newal
            old_policy[1]=newb
            if(old_policy[0]==-1):
                old_policy[0]=new_policy[0]
            if(old_policy[1]==-1):
                old_policy[1]=new_policy[0]
            
            # print(old_policy,new_policy)
            old_policy,new_policy=new_policy,old_policy
            #print(old_policy,new_policy)
    return (new_policy+[1,1])