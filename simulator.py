import numpy as np
from strategy import *
class Predictor:
    def __init__(self,prob,state,horizon,cost):
        self.pred=state
        self.prob=prob
        self.horizon=horizon
        self.cost=cost
        
        self.queried_state=state # updated automatically
        self.time_slots_elapsed=0 # updated automatically
        self.time_to_tau=0 #updated automatically

    def find_tau(self):
        # code to find tau and update it to self.time_to_tau
        # use cost + E[ Loss(tau to t with tau being latest queried state) ] <= E[ Loss(tau to t) ] 
        # self.time_to_tau=1
        # return
        t = 2
        while(1):
            print("@@@@@@       Searching tau for t = "+str(t)) 
            # A = E[ Loss(last queried to t) ] 
            # X = E[ Loss(last queried to tau) ] 
            # B = E[ Loss(tau to t) ]
            # C = E[ Loss(tau to t with tau being latest queried state) ]
            # thus inequality is cost + C <= B
            tau = 0
            temp_exp_loss = [0]
            for i in range(t):
                temp_exp_loss.append( temp_exp_loss[i] + np.min(np.matmul([1-self.queried_state, self.queried_state] , np.linalg.matrix_power(self.prob,i+1))))

            A = temp_exp_loss[t]
            for j in range(1,t): #j is being iterated to get tau thus j is temp_tau
                X = temp_exp_loss[j]
                B = A - X
                # state prob for tau
                temp = np.matmul([1-self.queried_state, self.queried_state] , np.linalg.matrix_power(self.prob,j))
                tau0_exp_loss = 0
                tau1_exp_loss = 0
                for k in range(t-j):
                    tau0_exp_loss += np.min(np.matmul([1, 0] , np.linalg.matrix_power(self.prob,k+1)))
                    tau1_exp_loss += np.min(np.matmul([0, 1] , np.linalg.matrix_power(self.prob,k+1)))
                C = np.dot(temp, [tau0_exp_loss, tau1_exp_loss])
                if self.cost + C <= B :
                    tau = j
                    break
            
            if tau != 0:
                self.time_to_tau = tau
                print("######       tau = "+str(tau)+" found for t = "+str(t)+" where C is "+str(C)+" and B is "+str(B))
                break
            t += 1
        return

    def reply_for_query(self, reply):
        self.queried_state = reply
        print("!!!!!!       State Queried and got "+str(reply))
        self.time_slots_elapsed=0
        self.find_tau()

    def predict(self):
        # Fill code here
        # To Probe, return -1
        if self.time_slots_elapsed + 1== self.time_to_tau: 
            print("??????       Queried for status")
            return -1
        # else
        temp_state = np.argmax(np.matmul([1-self.queried_state, self.queried_state] , np.linalg.matrix_power(self.prob,self.time_slots_elapsed+1)))
        print("######       Predicted state to be "+str(temp_state))
        return temp_state
        #note timeslot elapsed was not incremented thats why +1

class Simulator:
    def __init__(self, prob, state, horizon, cost):
        self.prob = prob
        self.state = state
        self.horizon = horizon
        self.cost = cost
        self.num = prob.shape[0]  #number of states
        self.history=[state]
    def next(self):
        tr=np.random.rand()
        
        tot=0.0
        
        for i in range(self.num):
            if tr<=tot+self.prob[self.state][i]:
                self.history.append(i)
                return i
            else:
                tot+=self.prob[self.state][i]

    def simulate(self):
        loss=0.0
        
        predictor = Predictor(self.prob,self.state,self.horizon,self.cost)
        predictor.reply_for_query(self.state) #sets up the predictor for first state provided
            
        for step in range(self.horizon):
            self.state = self.next()
            predicted_state = predictor.predict()
            
            if(predicted_state==-1):
                predictor.reply_for_query(self.state)
                loss+=self.cost
            else:
                predictor.time_slots_elapsed+=1
                loss+=(self.state-predicted_state)**2
        
        return loss,np.array(self.history)
            
if __name__=='__main__':

    prob = np.array([[0.9,0.1],[0.1,0.9]])  #[[p00, p01],[p10, p11]]
    # state 0 means s is [1, 0]  s*P = [p00, p01]
    # state 1 means s is [0, 1]  s*P = [p10, p11]
    sim = Simulator(prob,0,100,0.2)
    loss,history=sim.simulate()
    DP(history,prob,0.2,100)
    print("LOSS is = "+str(loss))