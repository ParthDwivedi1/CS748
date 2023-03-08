import numpy 
from strategy import *
from simulator import  Simulator
from constraint_optim import *

np.random.seed(1442)

def test_algo(algo_arr,test_cases=None):
    #tests against all the algoriths provided:: in the algo arr against THE conjectured algorithm::
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
    for test_case in test_cases:
            history_obj={}
            alpha,beta=optim_con(test_case['prob'],cost=test_case['cost'],Mx_val=100)
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
            write_file('test5.json',history_obj)

     
if __name__=='__main__':
    test_optim()


            