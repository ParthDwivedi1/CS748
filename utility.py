import numpy as np
import json
import logging
import datetime
logging.basicConfig(filename='/Users/saitejavaranasi/Desktop/rnd/CS748/LOGS/log'+str(datetime.datetime.now()),level=logging.DEBUG)
#numpy writer::
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
def loss_cal(prob,i,j,state): #i is position of last query, j is last state at which we need loss
    
    n=prob.shape[1]
    loss = np.zeros((n,1));
    # print(loss.shape)
    for x in range(j-i):
        temp = np.reshape(np.min(np.linalg.matrix_power(prob,x+1),axis=1),(2,1))
        # print(temp.shape)
        loss+= temp
    return np.matmul(state,loss)

def query_loss(i,j,res,prob):
        #this is a quey to return loss from i-j inclusive if a query is made given outcome
        state_curr=np.array([[1-res],[res]])
        loss=0
        for idx in range(i+1,j+1):
            state_curr = np.matmul(prob,state_curr)
            loss+=np.min(state_curr)
        # print("I: ",i,"J: ",j,"LOSS: ",loss)
        return loss

def write_file(fp,json_obj):
    json_=json.dumps(json_obj,indent=4,cls=NumpyEncoder)
    with open(fp,'a') as f:
        f.write(json_)

def write_log(log,type='info'):
    if(type=='info'):
        logging.info(log)
    elif(type=='debug'):
        logging.debug(log)
    elif(type=='warning'):
        logging.warning(log)
    elif(type=='error'):
        logging.error(log)
    elif(type=='critical'):
        logging.critical(log)


def read_ndjson(fp):
    json_str=""
    arr=[]
    with open(fp) as f:
        for line in f:
            if(len(line)==0):
                continue
            if((len(line)==1 and line[0]=='}') or (line[0]=='}' and line[1]=='{')):
                #print(json_str)
                json_str+='}'
                arr.append(json.loads(json_str))
                json_str="{"
            else:
                json_str+=line
    return arr