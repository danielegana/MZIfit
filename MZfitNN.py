#%%
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
from torch.utils.data import Dataset,random_split
import matplotlib.pyplot as plt
from numpy import random
import torch.nn.functional as F
from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import train_test_split
import torch.nn.init as init
import re
from scipy.interpolate import interp1d
import os
from sklearn.metrics import mean_squared_error
import itertools
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process
from itertools import starmap
import subprocess
from itertools import chain
import time
import joblib
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor



#%%

# Test func pool function

    #%%
pythondir="/Users/danielegana/MZIMLfiles/"
filedir="/Users/danielegana/MZIMLfiles/"
#%%
os.chdir(pythondir)
#%%

print("Defining parameters")

bwdtnm=50
bwdtmum=bwdtnm*1.0e-3
lendata=20000

maxnvar=0.1
maxnvar2=0.01
n0min=1.6
n0max=1.7
n1min=-maxnvar/bwdtmum
n1max=maxnvar/bwdtmum
n2min=-maxnvar2/bwdtmum**2
n2max=maxnvar2/bwdtmum**2
# these are the coefficients of the envelope fit around lambda-lambda0, in nm
maxoffset=50
lambda0max=-1
lambda0min=-10
lambda1max=maxoffset/bwdtnm
lambda1min=-maxoffset/bwdtnm
lambda2max=maxoffset/bwdtnm**2
lambda2min=-maxoffset/bwdtnm**2
lambda3max=maxoffset/bwdtnm**3
lambda3min=-maxoffset/bwdtnm**3
lambda4max=maxoffset/bwdtnm**4
lambda4min=-maxoffset/bwdtnm**4
train_size = int(0.8 * lendata)
val_size = lendata - train_size

# These are the 7 param tuples. In addition to the BH params, we have a random seed for different runs.
n0= np.random.permutation(np.linspace(n0min, n0max, lendata))
n1= np.random.permutation(np.linspace(n1min, n1max, lendata))
n2= np.random.permutation(np.linspace(n2min, n2max, lendata))
lambda0 = np.random.permutation(np.linspace(lambda0min, lambda0max, lendata))
lambda1 = np.random.permutation(np.linspace(lambda1min, lambda1max, lendata))
lambda2 = np.random.permutation(np.linspace(lambda2min, lambda2max, lendata))
lambda3 = np.random.permutation(np.linspace(lambda3min, lambda3max, lendata))
lambda4 = np.random.permutation(np.linspace(lambda4min, lambda4max, lendata))
randomseed=np.random.randint(0,1e9,lendata)
#%%

def TransferFdblist(lambdamum,lambda0mum,DeltaLmum,labelarray):
    dlambdamum=lambdamum-lambda0mum
    dlambdanm=dlambdamum*1.0e3
    neff=labelarray[0]+labelarray[1]*dlambdamum+labelarray[2]*dlambdamum**2
    offset=labelarray[3]+labelarray[4]*dlambdanm+labelarray[5]*dlambdanm**2+labelarray[6]*dlambdanm**3+labelarray[7]*dlambdanm**4
    beta=neff*2*np.pi/(lambdamum)
    #TFdB=10*np.log10(0.5*(1+np.cos(beta*DeltaLmum)))+2*offset
    TFdB=10*np.log10(0.5*(1+np.cos(beta*DeltaLmum)))
    return TFdB

class getDataset(Dataset):
    def __init__(self, labels, feature_functions,npointsfeaturefunc,DeltaLmum,lambda0mum):
        self.labels = labels
        self.feature_functions=feature_functions
        self.npointsfeaturefunc=npointsfeaturefunc
        self.DeltaLmum=DeltaLmum
        self.lambda0mum=lambda0mum

    def __len__(self):
        # Number of instances in the dataset
        return len(self.labels)

    def __getitem__(self, idx):
        # Select the label row at the given index
        labels = self.labels[idx]

        # Apply each feature function to the labels
        features = self.feature_functions(self.npointsfeaturefunc,self.lambda0mum,self.DeltaLmum,labels)

        # Concatenate the features if necessary

        # Return a tuple of (label, features)
        return labels, features

#%%
# labels=np.column_stack((n0,n1,n2,lambda0,lambda1,lambda2,lambda3,lambda4))
# labelsnorm=np.column_stack((n0/n0max,n1/n1max,n2/n2max,lambda0/maxoffset,lambda1/lambda1max,lambda2/lambda2max,lambda3/lambda3max,lambda4/lambda4max))

def neff(n0,n1,n2,dlambdamum):
    return n0+n1*dlambdamum+n2*dlambdamum**2

labels=np.column_stack((n0,n1,n2))
labels=np.array([row for row in labels if neff(row[0],row[1],row[2],bwdtmum) <= 1.7 and neff(row[0],row[1],row[2],bwdtmum) >= 1.6])
labelsnorm=np.column_stack((labels[:,0]/n0max,labels[:,1]/n1max,labels[:,2]/n2max))

#%%
labelben=np.array([1.67194,-0.324256,0.0678241,-5.507123133079745,-0.07110396303254388,-0.0024757350518301766,-0.00001588506651858012,-1.2176688398843063*10**(-7)])
labelbennorm=np.array([1.67194/n0max,-0.324256/n1max,0.0678241/n2max,-5.507123133079745/maxoffset,-0.07110396303254388/lambda1max,-0.0024757350518301766/lambda2max,-0.00001588506651858012/lambda3max,-1.2176688398843063*10**(-7)/lambda4max])

#%%
numpointslambda=2000
lambdamummin=1.26
lambdamu0=1.31
lambdamumax=1.36
lambdamum=np.linspace(lambdamummin,lambdamumax,numpointslambda)
#normalized_grid = (lambdamum - lambdamu0) / (lambdamumax - lambdamu0)
#transformed_grid = np.array([np.sqrt(x) if x>0 else -np.sqrt(-x) for x in normalized_grid])
#lambdamum = lambdamu0 + (lambdamumax - lambdamu0) * transformed_grid

#%%
DeltaL=151.47

#features=np.array([TransferFdblist(numpointslambda, 1.31, DeltaL,row) for row in labels])
#featuresben=np.array([TransferFdblist(numpointslambda, 1.31, DeltaL,labelben)])

features=np.array([TransferFdblist(lambdamum, 1.31, DeltaL,np.concatenate((row,np.array([labelben[3],labelben[4],labelben[5],labelben[6],labelben[7]])))) for row in labels])
featuresben=np.array([TransferFdblist(lambdamum, 1.31, DeltaL,labelben)])
X_train, X_test, y_train, y_test = train_test_split(features, labelsnorm, test_size=0.2, random_state=42)



#modelDT=ensemble.BaggingRegressor()

#%%
#modelDT=ensemble.RandomForestRegressor(n_jobs=-1,min_samples_leaf=1,max_depth=100,n_estimators=100)

modelDT=tree.DecisionTreeRegressor(max_depth=1000)
modelDT.fit(X_train,y_train)

y_pred = modelDT.predict(X_test)
y_predtrain = modelDT.predict(X_train)

#%%
frac_error=np.divide(y_predtrain,y_train)
mse = mean_squared_error(y_predtrain,y_train)
mae = np.sqrt(mean_squared_error(y_predtrain,y_train))
print("MFRE per feature",np.mean(frac_error, axis=0))
print("MSE DT: ",mse)
print("MAE DT: ",mae)

#%%
frac_error=np.divide(y_pred,y_test)
mse = mean_squared_error(y_test, y_pred)
mae = np.sqrt(mean_squared_error(y_test, y_pred))
print("MFRE per feature",np.mean(frac_error, axis=0))
print("MSE DT: ",mse)
print("MAE DT: ",mae)



#%%
#plt.plot(lambdamum,np.transpose(featuresben))

#%%
#plt.plot(lambdamum,np.transpose(features[3000]))
#%%
idxtest=40
testfun=TransferFdblist(lambdamum, 1.31, DeltaL,np.concatenate((y_predtrain[idxtest],np.array([labelben[3],labelben[4],labelben[5],labelben[6],labelben[7]]))))
plt.plot(lambdamum,np.transpose(testfun))
plt.plot(lambdamum,np.transpose(X_train[idxtest]),linestyle=':', color='blue')




    #%%
   # print("Accuracy:", accuracy_score(y_test[0], y_pred[0]))

    #%%


#%%