import numpy as np
from strategy import *
from utility import *
class Predictor:
    def __init__(self,prob,state,horizon,cost):
        self.pred=state
        self.prob=prob
        self.horizon=horizon
        self.cost=cost
        
        self.queried_state=state # updated automatically
        self.time_slots_elapsed=0 # updated automatically
        self.time_to_tau=0 #updated automatically
        self.LOSS_DICT=None
        self.tau_dict={}
    def find_tau(self,state):
        # code to find tau and update it to self.time_to_tau
        # use cost + E[ Loss(tau to t with tau being latest queried state) ] <= E[ Loss(tau to t) ] 
        # self.time_to_tau=1
        # return
        t = 2
        self.time_to_tau=float('inf')
        loss=float('inf')
        found=False#flag that tells if we found such tau::
        P=[1-state,state]#probabilities when we query at tau (can be randomly initialised)
        while(t<self.horizon):
            print("@@@@@@       Searching tau for t = "+str(t)) 
            # A = E[ Loss(last queried to t) ] 
            # X = E[ Loss(last queried to tau) ] 
            # B = E[ Loss(tau to t) ]
            # C = E[ Loss(tau to t with tau being latest queried state) ]
            # thus inequality is cost + C <= B
            tau = 0
            temp_exp_loss = [0]
            for i in range(t):
                temp_exp_loss.append( temp_exp_loss[i] + np.min(np.matmul([1-state, state] , np.linalg.matrix_power(self.prob,i+1))))

            A = temp_exp_loss[t]
            for j in range(1,t): #j is being iterated to get tau thus j is temp_tau
                X = temp_exp_loss[j-1]
                B = A - X
                # state prob for tau
                temp = np.matmul([1-state, state] , np.linalg.matrix_power(self.prob,j))
                tau0_exp_loss = 0
                tau1_exp_loss = 0
                for k in range(t-j):
                    tau0_exp_loss += np.min(np.matmul([1, 0] , np.linalg.matrix_power(self.prob,k+1)))
                    tau1_exp_loss += np.min(np.matmul([0, 1] , np.linalg.matrix_power(self.prob,k+1)))
                C = np.dot(temp, [tau0_exp_loss, tau1_exp_loss])
                #print("-----    ","Loss is",C," and loss is ",B)
                if self.cost + C <= B :
                    tau = j
                    loss=X
                    P=temp
                    found=True
                    break
            
            if tau != 0:
                self.time_to_tau = tau
                #print("######       tau = "+str(tau)+" found for t = "+str(t)+" where C is "+str(C)+" and B is "+str(B))

                break
            t += 1
        return {"found":found,"tau":tau,"loss":loss+self.cost,"P":P}

    def find_tau_old(self,state,T=1000):
        # code for alternate front solve(supposed to be optimal?)
        # T-> denotes the lookahead from current step we will be looking for to confirm if the value is tau:
        tau = 1
        self.time_to_tau=float('inf')
        loss=float('inf')
        found=False#flag that tells if we found such tau::
        P=[1-state,state]#probabilities when we query at tau (can be randomly initialised)
        if(self.LOSS_DICT is None):
            self.LOSS_DICT=[[0],[0]]
            prob_state=[1,0]
            loss=0
            for i in range(1,T+1):
                prob_state=np.matmul(prob_state,self.prob)
                loss+=np.min(prob_state)
                self.LOSS_DICT[0].append(loss)
            
            prob_state=[0,1]
            loss=0
            for i in range(1,T+1):
                prob_state=np.matmul(prob_state,self.prob)
                loss+=np.min(prob_state)
                self.LOSS_DICT[1].append(loss)
        while(tau<self.horizon):
            t = tau+1
            prob_state=np.matmul(P,np.power(self.prob,tau))
            prob_state_temp=prob_state
            B=np.min(prob_state_temp)
            while(t<=tau+T):
                print("@@@@@@       Searching t for tau = "+str(tau)) 
                # A = E[ Loss(last queried to t) ] 
                # X = E[ Loss(last queried to tau) ] 
                # B = E[ Loss(tau to t) ]
                # C = E[ Loss(tau to t with tau being latest queried state) ]
                # thus inequality is cost + C <= B
                
                
                prob_state_temp=np.matmul(prob_state_temp,self.prob)
                B+=np.min(prob_state_temp)
                
                C=prob_state[0]*self.LOSS_DICT[0][t-tau]+prob_state[1]*self.LOSS_DICT[1][t-tau]
                #X=P[0]*self.LOSS_DICT[0][tau-1]+P[1]*self.LOSS_DICT[1][tau-1]


                if(C+self.cost<=B):
                    found=True
                    loss=P[0]*self.LOSS_DICT[0][tau-1]+P[1]*self.LOSS_DICT[1][tau-1]
                    P=prob_state
                    break
                
                t += 1
            if(found):
                break
            tau+=1
        print({"found":found,"tau":tau,"loss":loss+self.cost,"P":P})
        return {"found":found,"tau":tau,"loss":loss+self.cost,"P":P}

    def reply_for_query(self, reply):
        self.queried_state = reply
        #print("!!!!!!       State Queried and got "+str(reply))
        self.time_slots_elapsed=0
        if(reply in self.tau_dict):
            self.time_to_tau=self.tau_dict[reply][1]
        else:
            tau_reply=self.find_tau(reply)
            self.tau_dict[reply]=[tau_reply['loss'],tau_reply["tau"],tau_reply["P"]]
    def get_tau(self):
        for i in range(2):
            if(i not in self.tau_dict):
                tau_reply=self.find_tau(i)
                self.tau_dict[i]=[tau_reply['loss'],tau_reply["tau"],tau_reply["P"]]
        return self.tau_dict
    def pred_exp_loss(self):
        for i in range(2):
            if(i not in self.tau_dict):
                tau_reply=self.find_tau(i)
                self.tau_dict[i]=[tau_reply['loss'],tau_reply["tau"],tau_reply["P"]]
        write_log(self.tau_dict)
        v_0=(self.tau_dict[0][0]*self.tau_dict[1][2][0]+self.tau_dict[0][2][1]*self.tau_dict[1][0])/(self.tau_dict[0][1]*self.tau_dict[1][2][0]+self.tau_dict[1][1]*self.tau_dict[0][2][1])
        v_1=(self.tau_dict[1][0]*self.tau_dict[0][2][1]+self.tau_dict[1][2][0]*self.tau_dict[0][0])/(self.tau_dict[1][1]*self.tau_dict[0][2][1]+self.tau_dict[0][1]*self.tau_dict[1][2][0])

        return [v_0,v_1],self.tau_dict
        
        

    def predict(self):
        # Fill code here
        # To Probe, return -1
        if self.time_slots_elapsed + 1== self.time_to_tau: 
            #print("??????       Queried for status")
            return -1
        # else
        temp_state = np.argmax(np.matmul([1-self.queried_state, self.queried_state] , np.linalg.matrix_power(self.prob,self.time_slots_elapsed+1)))
        #print("######       Predicted state to be "+str(temp_state))
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
        #write_log(f"tr:{tr},tot:{tot}")

    def simulate(self,tau_dict=None):
        loss=0.0
        
        predictor = Predictor(self.prob,self.state,self.horizon,self.cost)
        if(tau_dict is not None):
            predictor.tau_dict=tau_dict
        predictor.reply_for_query(self.state) #sets up the predictor for first state provided
            
        for step in range(self.horizon):
            self.state = self.next()
            predicted_state = predictor.predict()
            
            if(predicted_state==-1):
                predictor.reply_for_query(self.state)
                loss+=self.cost
            else:
                predictor.time_slots_elapsed+=1
                #write_log(f"num: {self.num}:::predicted_state: {predicted_state} ::: true_state:{self.state}",'error')
                loss+=(self.state-predicted_state)**2
        
        return loss,np.array(self.history)
            
if __name__=='__main__':

    prob = np.array([[0.8,0.2],[0.5,0.5]])  #[[p00, p01],[p10, p11]]
    # state 0 means s is [1, 0]  s*P = [p00, p01]
    # state 1 means s is [0, 1]  s*P = [p10, p11]
    loss1=0
    loss2=0

    for i in range(100):
        T=50
        cost=0.3
        pred=Predictor(prob,1,T,cost)
        arr,tau_=pred.pred_exp_loss()
        sim = Simulator(prob,1,T,cost)
        write_log(f"pred_loss {arr[0]*T-cost} {arr[1]*T-cost}")
        loss1_,history=sim.simulate()
        sim = Simulator(prob,0,T,cost)
        loss2_,history=sim.simulate()
        loss1+=loss1_
        loss2+=loss2_
        write_log(f"{i+1}/100 :{loss1/(i+1)} {loss2/(i+1)}")
    write_log(f"{loss1/(1000)} {loss2/(1000)}")