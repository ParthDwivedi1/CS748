import numpy as np

class Predictor:
    def __init__(self,prob,state,horizon,cost):
        self.pred=state
        self.prob=prob
        self.horizon=horizon
        self.cost=cost
        
        self.queried_state=state # updated automatically
        self.time_slots_elapsed=0 # updated automatically
    def predict(self):
        # Fill code here
        # To Probe, return -1
        # np.argmax(np.linalg.matrix_power(self.prob,self.time_slots_elapsed+1),axis=1)[0]
        return np.argmax(np.linalg.matrix_power(self.prob,self.time_slots_elapsed+1),axis=1)[0]

class Simulator:
    def __init__(self, prob, state, horizon, cost):
        self.prob = prob
        self.state = state
        self.horizon = horizon
        self.cost = cost
        self.num = prob.shape[0]
    
    def next(self):
        tr=np.random.rand()
        
        tot=0.0;
        
        for i in range(self.num):
            if tr<=tot+self.prob[self.state][i]:
                return i;
            else:
                tot+=self.prob[self.state][i]

    def simulate(self):
        loss=0.0
        
        predictor = Predictor(self.prob,self.state,self.horizon,self.cost)
            
        for step in range(self.horizon):
            self.state = self.next()
            predicted_state = predictor.predict();
            
            if(predicted_state==-1):
                predictor.queried_state=curr_state
                predictor.time_slots_elapsed=0
                loss+=self.cost
            else:
                predictor.time_slots_elapsed+=1
                loss+=(self.state-predicted_state)**2
        
        return loss
            
if __name__=='__main__':

    prob = np.array([[0.5,0.5],[0.5,0.5]])
    sim = Simulator(prob,0,10000,4)
    loss=sim.simulate()
    print(loss)