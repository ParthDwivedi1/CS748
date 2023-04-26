import numpy 
from strategy import *
from simulator import  Simulator,Predictor
from constraint_optim import *

np.random.seed(123)
def tc_dp_inf(n):
     p1=np.random.rand()
     p2=np.random.rand()
     cost=3*np.random.rand()
     test_cases=[]
     for i in range(n):
        dict_test_case={}
        dict_test_case['prob']=np.array([[p1,1-p1],[p2,1-p2]])
        dict_test_case['T']=10*(i+1)
        dict_test_case['cost']=cost
        test_cases.append(dict_test_case)
        write_log(f"Test case {i}/50 created")
     return test_cases
def test_algo(algo_arr,test_cases=None):
    #tests against all the algoriths provided:: in the algo arr against THE conjectured algorithm::
    if(test_cases is None):
        test_cases=[]
        for i in range(5):
            dict_test_case={}
            p1=np.random.rand()
            p2=np.random.rand()
            dict_test_case['prob']=np.array([[p1,1-p1],[p2,1-p2]])
            dict_test_case['T']=10*np.random.randint(1,20)
            dict_test_case['cost']=3*np.random.rand()
            test_cases.append(dict_test_case)
            write_log(f"Test case {i}/50 created")
     
    for test_case in test_cases:
            sim = Simulator(test_case['prob'],0,test_case['T'],test_case['cost'])
            loss,history=sim.simulate()
            #making a history obj that will be stored in a file:: for graphs and evaluation::
            history_obj={}
            history_obj['test_case']=test_case
            history_obj['history']=history
            
            loss_arr=[loss]
            #looping over all algorithms:: threading can be done here::
            for algo in algo_arr:
                loss_arr.append(algo(history,test_case['prob'],test_case['cost']))
            
            #saving losses in history__
            history_obj['loss']=loss_arr

            write_file('test2.json',history_obj)
            write_log(f"TEST CASE DONE::::")
def test_optim(test_cases=None):
    if(test_cases is None):
        test_cases=[]
        for i in range(50):
            dict_test_case={}
            p1=np.random.rand()
            p2=np.random.rand()
            dict_test_case['prob']=np.array([[p1,1-p1],[p2,1-p2]])
            dict_test_case['T']=10*np.random.randint(1,20)
            dict_test_case['cost']=3*np.random.rand()
            test_cases.append(dict_test_case)
            write_log(f"Test case {i}/50 created")
    
    co=0
    for test_case in test_cases:
            co+=1
            print(co)
            history_obj={}
            alpha,beta,v=optim_con(test_case['prob'],cost=test_case['cost'],Mx_val=100)
            loss=0
            for i in range(100):
                sim = Simulator(test_case['prob'],0,test_case['T'],test_case['cost'])
                loss_,history=sim.simulate({0:[0,alpha,0],1:[0,beta,0]})
                loss+=loss_
            loss/=100

            loss_=OPTPolicy(test_case['prob'],test_case['T'],test_case['cost'])
            history_obj['tc']=test_case
            history_obj['loss_infinite']=loss
            history_obj['loss_finite']=loss_
            write_file('test_dpvsinf_2.json',history_obj)
def test_optim_algo(test_cases=None):
    if(test_cases is None):
        test_cases=[]
        for i in range(100):
            dict_test_case={}
            p1=np.random.rand()
            p2=np.random.rand()
            dict_test_case['prob']=np.array([[p1,1-p1],[p2,1-p2]])
            dict_test_case['T']=10*np.random.randint(1,20)
            dict_test_case['cost']=3*np.random.rand()
            test_cases.append(dict_test_case)
            write_log(f"Test case {i}/100 created")
    for test_case in test_cases:
            history_obj={}
            alpha,beta,v=optim_con(test_case['prob'],cost=test_case['cost'],Mx_val=100)
            if(alpha==100 and beta==100):
                 continue
            pred= Predictor(test_case['prob'],0,100,test_case['cost'])
            loss,tau_=pred.pred_exp_loss()
            
            alpha_b,beta_b=BackwardSolve([0]*100,test_case['prob'],test_case['cost'])
            history_obj['tc']=test_case
            history_obj['loss_infinite']=v

            history_obj['tau_algo']=tau_
            history_obj['tau_infinite']={0:[0,alpha,0],1:[0,beta,0]}
            history_obj['tau_algo_back']={0:[0,alpha_b,0],1:[0,beta_b,0]}

            history_obj['loss_algo']=loss
            write_file('test_alogvsco_new_4.json',history_obj)
if __name__=='__main__':
    test_optim_algo()


            