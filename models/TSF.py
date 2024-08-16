#!/usr/bin/env python
# coding: utf-8

# In[1]:


from aeon.datasets import load_from_tsfile
from aeon.classification.interval_based import TimeSeriesForestClassifier
import numpy as np
import math

import model_utils
from model_utils import (modLog,Params,logging)

import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()


parser.add_argument('-data','--dataset_name',type = str,required=True)
parser.add_argument('-plots','--plots',type = bool,default = True)
parser.add_argument('-otb','--out_of_the_box',type = str,default = "y")


args = parser.parse_args()
dataset_name = args.dataset_name
plots = args.plots
otb = args.out_of_the_box


# In[22]:


class Node():
    def __init__(self,t1=None,t2=None,threshold=None,feature=None,left=None,right=None,info_gain=None,majority_class=None):
        self.tao = threshold #threshold 
        self.feature = feature #index of feature used to predict at this node
        self.leftChild = left
        self.rightChild = right
        self.IG = info_gain #information gain
        self.t1 = t1
        self.t2 = t2

        self.predClass = majority_class ##majority class of values at the leaf node


class TST(): #time series tree
    def __init__(self,k = 10, min_samples_split = 2,max_depth =2,flatten=True):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
       

        self.feats = {0:np.mean,1:np.std} ##map feature index to function
        self.Kappa = k

        self.flatten = flatten
    
        #initalize data with all samples at root  
    def compare(self,vec:np.ndarray,thresh:float):
        
        ##leq comparisono 
        if self.flatten:
            fnorm = np.linalg.norm(vec)
            if fnorm == thresh:
                return 0
            elif fnorm < thresh:
                return -1 
            else:
                return 1
        else:
            raise Exception("comparison not implemented for unflattened timesteps")

    def sample(self,M):
        T1,T2 = [],[]
        W = np.random.randint(1,M,int(math.sqrt(M)))
        for w in W:
            T1_temp = np.random.randint(1,M-w+1,int(math.sqrt(M-w+1)))
            for t1 in T1_temp: 
                end = t1+w-1
                if t1 not in T2 and end != t1:
                    T2.append(end)
                    T1.append(t1)
                    assert t1 != end , "interval of size 0"
        return [(t1,t2) for t1,t2 in zip(T1,T2)]

    def paritionTSDS(self,ds,t1,t2,feat,thresh):
        '''
        partition [t]ime [s]eries [d]ata [s]et

        ds: (ts,lbl) x Num samples 
        '''
        tup = [(x,y) for x,y in zip(ds[0],ds[1])]
        left =  [(ts,lbl) for ts,lbl in tup if self.compare(self.feats[feat](ts[:,t1:t2],axis=1),thresh) <= 0]  ## (dim,interval)
        right =  [(ts,lbl) for ts,lbl in tup if self.compare(self.feats[feat](ts[:,t1:t2],axis=1),thresh)  > 0]
        leftX = [ts for ts,_ in left]
        leftY = [lbl for _,lbl in left]
        rightX = [ts for ts,_ in right]
        rightY = [lbl for _,lbl in right]

        return [np.array(leftX),np.array(leftY)],[np.array(rightX),np.array(rightY)]
    def calculate_leaf_value(self,Y):
        Y = list(Y)
        return max(Y,key = Y.count)

    def makeTreeBatch(self,ds,curr_depth = 0):
        ''' 
        input: ds
        - X,y: (dim,seq_len)
        output:
        '''

        X,Y = ds[0],ds[1]
        num_feats,num_samples = X[0].shape
        if curr_depth <= self.max_depth and num_samples>=self.min_samples_split:
            best_split = self.get_best_split(ds,num_feats)
            if best_split["maxIG"] > 0:
                leftChild = self.makeTreeBatch(best_split['dL'],curr_depth +1)
                rightChild = self.makeTreeBatch(best_split['dR'],curr_depth +1)
                return Node(threshold = best_split["besttao"],
                            feature = best_split["bestfeat"],
                            left =leftChild,
                            right = rightChild,
                            info_gain= best_split["maxIG"],
                            t1 = best_split['bestt1'],
                            t2 = best_split['bestt2'])
        ## assert is_leaf
        leaf_value = self.calculate_leaf_value(Y)  
        return Node(majority_class = leaf_value)   

    def get_candidate_thresh(self,ds,t1,t2,flatten = True):
        '''
        output: (tree_features,sequence_features,kappa) OR (tree_features,kappa)

        '''
        X,Y = ds[0],ds[1]
        num_samples,seq_dim,seq_len = X.shape
        candidates_full = np.zeros((2,seq_dim,num_samples)) ##omitting slope from refrence paper for now 
        ##(tree_features,sequence_features,num_samples)
        for i,ts in enumerate(X):
            candidates_full[0,:,i] = np.mean(ts[:,t1:t2],axis = 1)
            candidates_full[1,:,i] = np.std(ts[:,t1:t2],axis = 1)

        

        candidates = np.zeros((2,seq_dim,self.Kappa))
        for i in range(seq_dim):
            for x in range(2):

                min = np.min(candidates_full[x,i,:])
                max = np.max(candidates_full[x,i,:])
                step = (max - min) / self.Kappa
                try:
                    candidates[x,i,:] = np.arange(start= min,
                                                    stop = max,
                                                    step= step)
                except ValueError: ##sometimes small intervals will cause repeat values, in which case range will be 0
                    pass
        
        if self.flatten:
            candidates_flat = np.zeros((2,self.Kappa))
            for i in range(self.Kappa):
                candidates_flat[0,i] = np.linalg.norm(candidates[0,:,i]) 
                candidates_flat[1,i] = np.linalg.norm(candidates[1,:,i]) 

        return candidates_flat # (tree_features,sequence_features,kappa) OR (tree_features,kappa)

    def get_entropy(self,y):
        ''' 
        input:
            y - list of class labels per sample. dim = (num_samples,)
        '''
        classes = np.unique(y)
        ylen =  len(y)
        entropy = 0
        for lbl in classes:
            class_prop = len(y[y == lbl]) / ylen
            entropy += -1 * class_prop * np.log2(class_prop) ##formula for entorpy
        return entropy

    def get_IG(self,parentY,lchildY,rchildY):
        wL = len(lchildY) / len(parentY)
        wR = len(rchildY) / len(parentY)
        return self.get_entropy(parentY) - (wL * self.get_entropy(lchildY) +wR * self.get_entropy(rchildY))

    def get_best_split(self,ds,num_feats):

        T = self.sample(ds[0].shape[-1]) ##double check that you can run sample here as opposed to other locations 


        info = {
            "maxIG":0, #delta Entropy 
            "bestt1": 0,
            "bestt2": 0,
            "besttao": 0,
            "bestfeat":None,
            "dL": None,
            "dR":None
        }
       
        
        for t1,t2 in T:
            assert t1 != t2 , "interval of size 0"
            for k in range(num_feats-1):
                #print("(t1,t2):",t1,":",t2)
    
                candidate_thresholds = self.get_candidate_thresh(ds,t1,t2) #f
    
                for tao in candidate_thresholds[k,:]:
                    dataLeft,dataRight = self.paritionTSDS(ds,t1,t2,k,tao)
                    IG = self.get_IG(ds[1],dataLeft[1],dataRight[1]) #delta entropy
                    if IG > info['maxIG']: 
                        info['maxIG'] = IG 
                        info['bestt1'] = t1 
                        info['bestt2'] = t2 
                        info['besttao'] = tao
                        info['bestfeat'] = k
                        info['dL'] = dataLeft
                        info['dR'] = dataRight


        assert (info['maxIG'] == 0) or (not np.array_equal(info['dL'],info['dR'])), 'left and right split are unepextecly equiv'
        ##either a leaft or they are equivalent
        return info
    def fit(self,X,Y):
        ds = [X,Y] ##tuple with form (time seires,lbabel)
        self.root = self.makeTreeBatch(ds)

    def predict(self,X):
        predictions = [self.make_prediction(x,self.root) for x in X]
        return predictions

    def make_prediction(self,ts,node):
        '''
        input: ts – (seuqence_dim,sequence_length)
        '''
        if node.predClass != None: 
            return node.predClass
        feature_val = self.feats[node.feature](ts[:,node.t1:node.t2],axis=1)
        cmp = self.compare(feature_val,node.tao)
        if cmp <= 0: ##could deal with this by overwritting less than
            return self.make_prediction(ts,node.leftChild)
        else:
            return self.make_prediction(ts,node.rightChild)

    def print_tree(self,curr_node = None, indent=" "):
        if curr_node is None:
            curr_node = self.root
        
        if curr_node.predClass is not None:
            print(curr_node.predClass)
        
        else:
            print(f"f_{curr_node.feature}({curr_node.t1},{curr_node.t2}) <= {curr_node.tao}")
            print("%sleft:" % (indent),end="")
            self.print_tree(curr_node.leftChild,indent+indent)
            print("%sright:" % (indent),end="")
            self.print_tree(curr_node.rightChild,indent+indent)


class TSF():
    def __init__(self,kappa=10,max_depth=3,min_sample_split=2,flatten=True,num_trees = 10):
        self.ensemble = {k:TST(kappa,min_sample_split,max_depth,flatten) for k in range(num_trees)}
        self.num_trees = num_trees
        self.lbl_type = None
    
    def fit(self,X,Y):
        self.lbl_type = Y[0].dtype
        for k,tree in self.ensemble.items():
            tree.fit(X,Y)
            self.ensemble[k] = tree
    
    def predict(self,X):
        num_samples = X.shape[0]
        predictions_full = np.empty((self.num_trees,num_samples),dtype=self.lbl_type) ##(trees,samples)
        for k,tree in self.ensemble.items():
            predictions_full[k,:] = tree.predict(X)
        votes = np.empty(num_samples,dtype=self.lbl_type)
        for i in range(num_samples):
            classes,counts = np.unique(predictions_full[:,i],return_counts=True)
            votes[i] = classes[np.argmax(counts)]
        return votes
    
    def score(self,preds,targets):
        classes, counts = np.unique(preds == targets,return_counts = True)
        acc = {c:count/np.sum(counts) for c,count in zip(classes,counts)}
        return acc[True]


# In[ ]:

def predict_and_score(tsf,X,Y):
        predictions = tsf.predict(X)
        classes,counts = np.unique(predictions==Y,return_counts=True)
        acc = {c:count/np.sum(counts) for c,count in zip(classes,counts)}
        return acc[True] ##True corresponds to when predictions and actual values are equal

def get_data(dataset_name):
    x,y = load_from_tsfile(f"./data/{dataset_name}/raw/{dataset_name}_TRAIN.ts")
    x_test,y_test = load_from_tsfile(f"./data/{dataset_name}/raw/{dataset_name}_TEST.ts")
    return x,y,x_test,y_test

def get_model(otb=True,n_trees=250,candidate_samples=10,max_depth=3,min_sample_split =2): 
    if otb == "y": 
        return TimeSeriesForestClassifier(n_estimators=n_trees,
                                 time_limit_in_minutes = 1)
    elif otb == "n": 
        return TSF(num_trees = n_trees,
                   kappa = candidate_samples,
                   max_depth = max_depth,
                   min_sample_split = min_sample_split)

if __name__ == "__main__":
    params = Params("./tools/tsf/params.json").params
    modLog("tsf",params)
    logging.info(f"========    LOADING DATASET : {dataset_name}    ======== ")
    X,Y,X_test,Y_test = get_data(dataset_name)
    TSFC = get_model(otb=otb,
                     n_trees = params["ntrees"],
                     candidate_samples = params["kappa"],
                     max_depth = params["depth"],
                     min_sample_split = params["minsplit"])
    TSFC.fit(X,Y)

    acc = predict_and_score(TSFC,X_test,Y_test)
    logging.info(f"======================================================= ")
    print(f"FINAL ACCURACY: {acc}")
    logging.info(f"FINAL ACCURACY: {acc}")

    if plots:
        cm = confusion_matrix(TSFC.predict(X_test),Y_test)
        cm = ConfusionMatrixDisplay(cm)
                    
        cm.plot()
        plt.xticks([])
        plt.yticks([])
        os.makedirs(f'./plots/{dataset_name}/confusion_matrix/tsf/classification',exist_ok=True)
        plt.savefig(f'./plots/{dataset_name}/confusion_matrix/tsf/classification/cm.png')
    

# SOURCE: 
# - https://arxiv.org/pdf/1302.2277
# - https://www.youtube.com/watch?v=sgQAhG5Q7iY