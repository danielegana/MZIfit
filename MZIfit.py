#%%
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
from torch.utils.data import Dataset,random_split
from numpy iPmport random
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
from scipy.signal import find_peaks
import lightgbm as lgb
import math
from sklearn.svm import SVR

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcdefaults()
from matplotlib import font_manager
from matplotlib import rcParams
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter, LogLocator
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
rcParams['mathtext.rm'] = 'Times New Roman'
rcParams['text.usetex'] = True
rcParams['font.family'] = 'times' #'sans-serif'
font_manager.findfont('serif', rebuild_if_missing=True)
rcParams.update({'font.size':14})


#%%
pythondir="/Users/danielegana/MZIMLfiles/"
filedir="/Users/danielegana/MZIMLfiles/"
#%%
os.chdir(pythondir)
#%%
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
#%%
#Define effective index functions

def neff(n0,n1,n2,dlambdamum):
    return n0+n1*dlambdamum+n2*dlambdamum**2

def neffoverlambdaabc(na,nb,nc,dlambdamum): #na, nb and nc are the series coefficients of neff/lambda
    return na+nb*dlambdamum+nc*dlambdamum**2

def toabc(n0,ng,Disp,lambdamum0):
    na=n0/lambdamum0
    nb=-ng/lambdamum0**2
    nc=(-1/2*Disp/lambdamum0-nb)/lambdamum0
    return np.array([na,nb,nc])

def tongfromabc(na,nb,nc,lambdamum0): #Transform between na,nb,nc and n0, ng and Dispersion
    n0=na*lambdamum0
    ng=-nb*lambdamum0**2
    Disp=-2*lambdamum0*(nb + nc*lambdamum0)
    return np.array([n0,ng,Disp])

def tongfromabclabel(labelarray,lambdamum0): #Same, but taking as inputs an array
    n0=labelarray[0]*lambdamum0
    ng=-labelarray[1]*lambdamum0**2
    Disp=-2*lambdamum0*(labelarray[1] + labelarray[2]*lambdamum0)
    return np.array([n0,ng,Disp,labelarray[3],labelarray[4],labelarray[5],labelarray[6],labelarray[7]])

def tongfromabclabelshort(labelarray,lambdamum0): #Same, but taking as inputs an array
    n0=labelarray[0]*lambdamum0
    ng=-labelarray[1]*lambdamum0**2
    Disp=-2*lambdamum0*(labelarray[1] + labelarray[2]*lambdamum0)
    return np.array([n0,ng,Disp])


#%%

print("Defining parameters")

bwdtnm=50
bwdtmum=bwdtnm*1.0e-3
lendata=3000000  #Number of training curves


#### 

#%%
DeltaL=151.47

#Lambda array definition
numpointslambda=51
lambdamummin=1.26
lambdamum0=1.31
lambdamummax=1.36
lambdamum=np.linspace(lambdamummin,lambdamummax,numpointslambda)
dlambdamum=lambdamum-lambdamum0
dlambdadist=dlambdamum[1]-dlambdamum[0]

numpointslambdaHR=301
lambdamumHR=np.linspace(lambdamummin,lambdamummax,numpointslambdaHR)
dlambdamumHR=lambdamumHR-lambdamum0

maxnvar=0.1
maxnvar2=0.01
n0min=1.6
n0max=1.7
n1min=-maxnvar/bwdtmum
n1max=maxnvar/bwdtmum
n2min=-maxnvar2/bwdtmum**2
n2max=maxnvar2/bwdtmum**2
# these are the coefficients of the envelope fit around lambda-lambda0, in nm
maxoffset=20
off0max=-1
off0min=-10
off1max=maxoffset/bwdtnm
off1min=-maxoffset/bwdtnm
off2max=maxoffset/bwdtnm**2
off2min=-maxoffset/bwdtnm**2
off3max=maxoffset/bwdtnm**3
off3min=-maxoffset/bwdtnm**3
off4max=maxoffset/bwdtnm**4
off4min=-maxoffset/bwdtnm**4

n0= np.random.permutation(np.linspace(n0min, n0max, lendata))
n1= np.random.permutation(np.linspace(n1min, n1max, lendata))
n2= np.random.permutation(np.linspace(n2min, n2max, lendata))

na=n0/lambdamum0
nb=(n1*lambdamum0-n0)/(lambdamum0**2) #This is minus the group index
nc=(n0-n1*lambdamum0+n2*lambdamum0**2)/(lambdamum0**3)

ngarray=(n0-n1*lambdamum0)
Disparray=-2*lambdamum0*n2

off0 = np.random.permutation(np.linspace(off0min, off0max, lendata))
off1 = np.random.permutation(np.linspace(off1min, off1max, lendata))
off2 = np.random.permutation(np.linspace(off2min, off2max, lendata))
off3 = np.random.permutation(np.linspace(off3min, off3max, lendata))
off4 = np.random.permutation(np.linspace(off4min, off4max, lendata))

dlambdanm=dlambdamum*1.0e3
fulloffset=[off0[idx]+off1[idx]*dlambdanm+off2[idx]*dlambdanm**2+off3[idx]*dlambdanm**3+off4[idx]*dlambdanm**4 for idx in range(lendata)]

#%%
labels=np.column_stack((na,nb,nc,off0,off1,off2,off3,off4))

labels = np.array([
    labels[idx] for idx in range(len(labels))
    if (
        1.6 <= lambdamum0 * neffoverlambdaabc(labels[idx][0], labels[idx][1], labels[idx][2], bwdtmum) <= 1.7
        and np.abs((labels[idx][1] * DeltaL) ** -1) < bwdtmum / 2
        and labels[idx][1] < 0
        and np.all(fulloffset[idx] < 0)
    )
])

#%%
nabase=labels[:,0]
nap=(2*np.pi*labels[:,0]*DeltaL % (math.tau/2))/(2*np.pi*DeltaL)
labels=np.column_stack((nap,labels[:,1],labels[:,2],labels[:,3],labels[:,4],labels[:,5],labels[:,6],labels[:,7]))

labelsdisp=np.array([tongfromabclabel(labelarray,lambdamum0) for labelarray in labels])

train_size = int(0.8 * labels.shape[0])
val_size = labels.shape[0] - train_size

# labels=np.column_stack((n0,n1,n2,off0,off1,off2,off3,off4))

#%% Define Transfer Function


def TransferFdblist(dlambdamum,DeltaLmum,labelarray):
    dlambdanm=dlambdamum*1000
    offset=labelarray[3]+labelarray[4]*dlambdanm+labelarray[5]*dlambdanm**2+labelarray[6]*dlambdanm**3+labelarray[7]*dlambdanm**4
    beta=neffoverlambdaabc(labelarray[0],labelarray[1],labelarray[2],dlambdamum)*2*np.pi
    TFdB=10*np.log10(0.5*(1+np.cos(beta*DeltaLmum)))+2*offset
   # TFdB=10*np.log10(0.5*(1+np.cos(beta*DeltaLmum)))
    return TFdB

def FSR(Tlist,dlambdadist,lambdamum,lambdamum0): # Free spectral range
    peaks=find_peaks(-Tlist)[0]
    lambdapeaks=lambdamum[peaks]
    centralindex=np.argmin(np.abs(lambdapeaks-lambdamum0))
    return  peaks,centralindex,lambdapeaks,np.diff(peaks*dlambdadist)  #The last one is the FSR


def allpeaks(Tlist,dlambdadist,lambdamum,lambdamum0): #Finds all the peaks in the function
    peaksdown=find_peaks(-Tlist)[0]
    peaksup=find_peaks(Tlist)[0]
    lambdapeaks=np.concatenate((lambdamum[peaksdown],lambdamum[peaksup]))
    return  lambdapeaks

def ngFSR(Tlist,dlambdadist,lambdamum,lambdamum0,DeltaLmum):
    _,centralindex,lambdapeaks,deltalambdapeaks=FSR(Tlist,dlambdadist,lambdamum,lambdamum0)
    lambdaav=(lambdapeaks[:-1]+lambdapeaks[1:])/2
    ng=lambdaav**2/(deltalambdapeaks*DeltaLmum)
    ngcentral=ng[centralindex]
    Disp,_=np.polyfit(lambdaav,ng,1)
    return lambdapeaks,ng,ngcentral,Disp #Finds the group index from the FSR


#%%
#Let's see if the group index from the FSR is correct, with an example
indextest=5
ngtest=ngFSR(TransferFdblist(dlambdamum, DeltaL,labels[indextest]),dlambdadist,lambdamum,lambdamum0,DeltaL)[2]
ngtrue=tongfromabclabel(labels[indextest],lambdamum0)[1]
print("Error in ng is: ",(ngtest-ngtrue)/ngtrue)
#%%
plt.plot(lambdamum,TransferFdblist(dlambdamum, DeltaL,labels[indextest]),color='b')
plt.vlines(ngFSR(TransferFdblist(dlambdamum, DeltaL,labels[indextest]),dlambdadist,lambdamum,lambdamum0,DeltaL)[0], ymin=-50, ymax=0, color='r', linestyle='--')
plt.plot(lambdamum,TransferFdblist(dlambdamum, DeltaL,labels[indextest]))

#%%

# First-parameter fit: n0

targetfunctionT=[TransferFdblist(dlambdamum, DeltaL,row) for row in labels]
targetfunction=[allpeaks(TransferFdblist(dlambdamum, DeltaL,labels[idx]),dlambdadist,lambdamum,lambdamum0) for idx in range(len(labels))]
# Determine the maximum length of the arrays
max_length = max(len(arr) for arr in targetfunction)
# Pad each array to the maximum length with zeros
targetfunction = [np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in targetfunction] 
features=np.array(targetfunction)
X_train, X_test, y_train, y_test = train_test_split(features, labels[:,0], test_size=0.4, random_state=30)
X_trainT, X_testT,y_trainfull, y_testfull = train_test_split(np.array(targetfunctionT), labelsdisp, test_size=0.4, random_state=30)
_,_,_,nabasetest = train_test_split(np.array(targetfunctionT), nabase, test_size=0.4, random_state=30)
## y_testfull contains the physical parameters, phase, ng, D, not na,nb,nc. I train on na,nb,nc though


#%%
# Train the model

#%%
train_data = lgb.Dataset(X_train, label=y_train)
params = {'objective': 'regression', 'metric': 'mse'}
baseline_params = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "seed": 42,
}
lr_params = baseline_params.copy()
modelDT1 = lgb.train(lr_params, train_data, num_boost_round=3000)


y_pred = modelDT1.predict(X_test)
y_predtrain = modelDT1.predict(X_train)

#%%
frac_error=np.divide(y_predtrain,y_train)
mse = mean_squared_error(y_predtrain,y_train)
mae = np.sqrt(mean_squared_error(y_predtrain,y_train))
print("MFRE per feature",np.mean(frac_error, axis=0))
print("MSE DT: ",mse)
print("MAE DT: ",mae)

#%%
y_pred=(2*np.pi*y_pred*DeltaL % (math.tau/2))/(2*np.pi*DeltaL) #Make sure na is predicted modulo Pi

frac_error=np.divide(y_pred,y_test)
mse = mean_squared_error(y_test, y_pred)
mae = np.sqrt(mean_squared_error(y_test, y_pred))
print("MFRE per feature",np.mean(frac_error, axis=0))
print("MSE DT: ",mse)
print("MAE DT: ",mae)
#%%
#Test the results
indextest=2
ngtest=ngFSR(X_testT[indextest],dlambdadist,lambdamum,lambdamum0,DeltaL)[2]
plt.plot(lambdamum,X_testT[indextest],color='b')
plt.vlines(ngFSR(X_testT[indextest],dlambdadist,lambdamum,lambdamum0,DeltaL)[0], ymin=-50, ymax=0, color='r', linestyle='--')
predarray=np.append(toabc(y_pred[indextest]*lambdamum0,ngtest,0,lambdamum0),[y_testfull[indextest][3],y_testfull[indextest][4],y_testfull[indextest][5],y_testfull[indextest][6],y_testfull[indextest][7]])
plt.plot(lambdamum,TransferFdblist(dlambdamum, DeltaL,predarray),color='g')

#Write out the predicted test labels. Note we set the dispersion to zero.
predna=modelDT1.predict(targetfunction)

counter=0
predng = []
for Testfun in targetfunctionT:
    try:
        result = ngFSR(Testfun, dlambdadist, lambdamum, lambdamum0, DeltaL)
        predng.append(result[2])  # Append the third element if successful
    except Exception as e:
        # Optionally, print or log the error if you need to know what failed
       # print(f"Error processing test")
        predng.append(0)
        counter+=1
predng=np.array(predng)

predlabels1=[toabc(predna[idx]*lambdamum0,predng[idx],0,lambdamum0) for idx in range(len(predna))]
#%%
##########

#%% Second parameter fit: Dispersion. We use the same target function as above as features
#%%
X_train2, X_test2, y_train2, y_test2 = train_test_split(features, labels[:,2], test_size=0.4, random_state=30)
_,_,_,Disparraytest=train_test_split(features, labelsdisp[:,2], test_size=0.4, random_state=30)

#%%
#Let's now use lgboost, it gives better results
train_data = lgb.Dataset(X_train2, label=y_train2)
params = {'objective': 'regression', 'metric': 'mse'}
baseline_params = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "seed": 42,
}
lr_params = baseline_params.copy()
modelDT2 = lgb.train(lr_params, train_data, num_boost_round=3000)
#%%

y_pred2 = modelDT2.predict(X_test2)
y_predtrain2 = modelDT2.predict(X_train2)

#%%
frac_error=np.divide(y_predtrain2,y_train2)
mse = mean_squared_error(y_predtrain2,y_train2)
mae = np.sqrt(mean_squared_error(y_predtrain2,y_train2))
print("MFRE per feature",np.mean(frac_error, axis=0))
print("MSE DT: ",mse)
print("MAE DT: ",mae)

#%%

frac_error=np.divide(y_pred2,y_test2)
relerror=np.divide(y_pred2-y_test2,y_pred2)
mse = mean_squared_error(y_test2, y_pred2)
mae = np.sqrt(mean_squared_error(y_test2, y_pred2))
print("MFRE per feature",np.mean(frac_error, axis=0))
print("MSE DT: ",mse)
print("MAE DT: ",mae)
#%%

indextest=2
_,_,ngtest,Disptest=ngFSR(X_testT[indextest],dlambdadist,lambdamum,lambdamum0,DeltaL)
plt.plot(lambdamum,X_testT[indextest],color='b')
plt.vlines(allpeaks(X_testT[indextest],dlambdadist,lambdamum,lambdamum0), ymin=-10, ymax=0, color='r', linestyle='--')
predarray=np.append(toabc(y_pred[indextest]*lambdamum0,ngtest,y_pred2[indextest],lambdamum0),[y_testfull[indextest][3],y_testfull[indextest][4],y_testfull[indextest][5],y_testfull[indextest][6],y_testfull[indextest][7]])
#predarray=toabc(y_pred[indextest]*lambdamum0,ngtest,y_pred2[indextest],lambdamum0)
Disppred=tongfromabclabel(predarray,lambdamum0)[2]
plt.plot(lambdamum,TransferFdblist(dlambdamum, DeltaL,predarray),color='g')
print("True disp = %.3f, From Slope = %.3f, Disppred= %.3f" %(Disparraytest[indextest], Disptest,Disppred))

#%% Third parameter fit: ng. We use the same target function as above as features
#%%
X_train3, X_test3, y_train3, y_test3 = train_test_split(features, labels[:,1], test_size=0.4, random_state=30)
_,_,_,ngarraytest=train_test_split(features, labelsdisp[:,1], test_size=0.4, random_state=30)

#%%
#Let's now use lgboost, it gives better results
train_data = lgb.Dataset(X_train3, label=y_train3)
params = {'objective': 'regression', 'metric': 'mse'}
baseline_params = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "seed": 42,
}
lr_params = baseline_params.copy()
modelDT3 = lgb.train(lr_params, train_data, num_boost_round=3000)
#%%

y_pred3 = modelDT3.predict(X_test3)
y_predtrain3 = modelDT3.predict(X_train3)

#%%
frac_error=np.divide(y_predtrain3,y_train3)
mse = mean_squared_error(y_predtrain3,y_train3)
mae = np.sqrt(mean_squared_error(y_predtrain3,y_train3))
print("MFRE per feature",np.mean(frac_error, axis=0))
print("MSE DT: ",mse)
print("MAE DT: ",mae)

#%%
frac_error=np.divide(y_pred3,y_test3)
mse = mean_squared_error(y_test3, y_pred3)
mae = np.sqrt(mean_squared_error(y_test3, y_pred3))
print("MFRE per feature",np.mean(frac_error, axis=0))
print("MSE DT: ",mse)
print("MAE DT: ",mae)
#%%

#Check how well the DTs fit the data


predarrayabc=[np.array([y_pred[idx],y_pred3[idx],y_pred2[idx]]) for idx in range(len(y_pred))]
predarray=np.array([tongfromabclabelshort(predarrayabc[idx],lambdamum0) for idx in range(len(y_pred))])

#%%


counter=0
predngtestlectures = []
for Testfun in X_testT:
    try:
        result = ngFSR(Testfun, dlambdadist, lambdamum, lambdamum0, DeltaL)
        predngtestlectures.append(result[2])  # Append the third element if successful
    except Exception as e:
        # Optionally, print or log the error if you need to know what failed
        print(f"Error processing test")
        predngtestlectures.append(0)
        counter+=1 #I check that not many of the test examples fail. It's bc findpeaks doesn't find peaks there!
predngtestlectures=np.array(predngtestlectures)


#%%
predDisptestlectures = []
for Testfun in X_testT:
    try:
        result = ngFSR(Testfun, dlambdadist, lambdamum, lambdamum0, DeltaL)
        predDisptestlectures.append(result[3])  # Append the third element if successful
    except Exception as e:
        # Optionally, print or log the error if you need to know what failed
        print(f"Error processing test")
        predDisptestlectures.append(0)
predDisptestlectures=np.array(predDisptestlectures)

predtestlecturesabc=np.array([toabc(1/2*lambdamum0/DeltaL,predngtestlectures[idx],predDisptestlectures[idx],lambdamum0) for idx in range(len(X_testT))])

#%%
fig, axes = plt.subplots(1, 3,figsize=(10, 8))  # (rows, cols)
axes[0].plot(2*np.pi*y_test*DeltaL,2*np.pi*predarray[:,0]/lambdamum0*DeltaL, marker='o',linestyle='None', markersize=1)
residuals=2*np.pi*y_test*DeltaL-2*np.pi*predarray[:,0]/lambdamum0*DeltaL
plt.text(-65, 8, "$\sigma$ = %.3f" % np.std(residuals),fontsize=12, color='blue')
axes[0].set_title("$\phi$")
axes[0].set_aspect('equal')  # Make it square
axes[0].set_xlabel('True')
axes[0].set_ylabel('Predicted')
# Second panel
axes[1].plot(y_testfull[:,1],predarray[:,1], marker='o',linestyle='None', markersize=1)
axes[1].set_title("$n_g$")
axes[1].set_aspect('equal')  # Make it square
residuals=y_testfull[:,1]-predarray[:,1]
plt.text(-35, 8, "$\sigma$ = %.3f" % np.std(residuals),fontsize=12, color='blue')
axes[1].set_xlabel('True')
axes[1].set_ylabel('Predicted')
# Third panel
axes[2].plot(y_testfull[:,2],predarray[:,2], marker='o',linestyle='None', markersize=1)
axes[2].set_title("$D$ $[\mu\mathrm{m}^{-1}]$")
axes[2].set_aspect('equal')  # Make it square
axes[2].set_xlabel('True')
axes[2].set_ylabel('Predicted')
residuals=y_testfull[:,2]-predarray[:,2]
plt.text(-10, 8, "$\sigma$ = %.3f" % np.std(residuals),fontsize=12, color='blue')
#%%
#Now perform a comparison with the methods suggested in the lecture
fig, axes = plt.subplots(1, 2)  # (rows, cols)
# First panel
axes[0].plot(y_testfull[:,1],predngtestlectures, marker='o',linestyle='None', markersize=1)
axes[0].set_title("ng")
axes[0].set_aspect('equal')  # Make it square
residuals=y_testfull[:,1]-predngtestlectures
plt.text(-35, 8, "$\sigma$ = %.3f" % np.std(residuals),fontsize=12, color='blue')
# Second panel?"[=-0p;..lp0-6
# .00-800000008]"
axes[1].plot(y_testfull[:,2],predDisptestlectures, marker='o',linestyle='None', markersize=1)
axes[1].set_title("Disp")
residuals=y_testfull[:,2]-predDisptestlectures
plt.text(-10, 8, "$\sigma$ = %.3f" % np.std(residuals),fontsize=12, color='blue')
axes[1].set_aspect('equal')  # Make it square

######## INDIVIDUAL PLOTS FOR REPORT ########
## PHASE
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
axes.plot(2*np.pi*y_test*DeltaL,2*np.pi*predarray[:,0]/lambdamum0*DeltaL, marker='o',linestyle='None', markersize=0.5,color='black')
residuals=2*np.pi*y_test*DeltaL-2*np.pi*predarray[:,0]/lambdamum0*DeltaL
plt.text(np.pi/2-0.3, 3, "$\sigma$ = %.3f" % np.std(residuals),fontsize=14, color='black')
axes.set_title("$\phi$")
#axes.set_aspect('equal')  # Make it square
axes.set_xlabel('True')
axes.set_ylabel('Predicted')
fig.tight_layout()
fig.savefig('/Users/danielegana/Dropbox (Personal)/Si Photonics/report/figures/phasetree.pdf', bbox_inches="tight")
#%%
## ng
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
axes.plot(y_testfull[:,1],predarray[:,1], marker='o',linestyle='None', markersize=1,color='black')
residuals=y_testfull[:,1]-predarray[:,1]
plt.text(1.4, 2.8, "$\sigma$ = %.3f" % np.std(residuals),fontsize=14, color='black')
axes.set_title("$n_g$")
#axes.set_aspect('equal')  # Make it square
axes.set_xlabel('True')
axes.set_ylabel('Predicted')
fig.tight_layout()
fig.savefig('/Users/danielegana/Dropbox (Personal)/Si Photonics/report/figures/ngtree.pdf', bbox_inches="tight")
#%%
## Disp
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
axes.plot(y_testfull[:,2],predarray[:,2], marker='o',linestyle='None', markersize=1,color='black')
axes.set_title("$D$ $[\mu\mathrm{m}^{-1}]$")
residuals=y_testfull[:,2]-predarray[:,2]
plt.text(-2.5,11, "$\sigma$ = %.3f" % np.std(residuals),fontsize=14, color='black')
#axes.set_aspect('equal')  # Make it square
axes.set_xlabel('True')
axes.set_ylabel('Predicted')
fig.tight_layout()
fig.savefig('/Users/danielegana/Dropbox (Personal)/Si Photonics/report/figures/disptree.pdf', bbox_inches="tight")
#%%

## Classicals 
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
axes.plot(y_testfull[:,1],predngtestlectures, marker='o',linestyle='None', markersize=1,color='black')
axes.set_title("$n_g$")
residuals=y_testfull[:,1]-predngtestlectures
plt.text(1.3, 2.75, "$\sigma$ = %.3f" % np.std(residuals),fontsize=14, color='black')
#axes.set_aspect('equal')  # Make it square
axes.set_xlabel('True')
axes.set_ylabel('Predicted')
fig.tight_layout()
plt.plot()
fig.savefig('/Users/danielegana/Dropbox (Personal)/Si Photonics/report/figures/ngfclass.pdf', bbox_inches="tight")

#%%
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
axes.plot(y_testfull[:,2],predDisptestlectures, marker='o',linestyle='None', markersize=1,color='black')
axes.set_title("$D$ $[\mu\mathrm{m}^{-1}]$")
residuals=y_testfull[:,2]-predDisptestlectures
plt.text(-2.5, 14, "$\sigma$ = %.3f" % np.std(residuals),fontsize=14, color='black')
#axes.set_aspect('equal')  # Make it square
axes.set_xlabel('True')
axes.set_ylabel('Predicted')
fig.tight_layout()
fig.savefig('/Users/danielegana/Dropbox (Personal)/Si Photonics/report/figures/dispclass.pdf', bbox_inches="tight")

#%% Now the intensity
#na=n0/lambdamum0
#nb=(n1*lambdamum0-n0)/(lambdamum0**2) #This is minus the group index
#nc=(n0-n1*lambdamum0+n2*lambdamum0**2)/(lambdamum0**3)

naben=1.67194/lambdamum0
nbben=((-0.324)*lambdamum0-1.67194)/(lambdamum0**2)
ncben=(1.67194-((-0.324))*lambdamum0+(-0*0.0678)*lambdamum0**2)/(lambdamum0**3)
offsetbenlambda=-327053.27192927484+1.0196063778848967e6*lambdamum-1.1938323213090852e6*lambdamum**2 +622173.4055807965*lambdamum**3 - 121766.88398843064*lambdamum**4
off0ben=-5.5071231330797
off1ben=-0.07110396303254388
off2ben=-2.4757350518301766e-3
off3ben=-0.00001588506651858012
off4ben=-1.2176688398843063*10**(-7)
offsetbendlambda=-5.5071231330797-71.10396303254388*dlambdamum-2.4757350518301766e3*dlambdamum**2-0.00001588506651858012e9*dlambdamum**3-1.2176688398843063*10**(5)*dlambdamum**4
#%%
indextest=10


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5,7))
#axes[0].plot(lambdamum,X_testT[indextest],color='black')
predtrue=np.array([naben,nbben,ncben,off0ben,off1ben,off2ben,off3ben,off4ben])

targetforplot=np.array(TransferFdblist(dlambdamum, DeltaL,predtrue))
#axes[0].vlines(allpeaks(targetforplot,dlambdadist,lambdamum,lambdamum0), ymin=-10, ymax=0, color='r', linestyle='--')
axes[0].plot(lambdamum,targetforplot,color='black',linewidth=0.8)
axes[0].set_ylabel('$F_{\mathrm{measured}}$')
axes[0].set_xticklabels([])

targetformodel=allpeaks(targetforplot,dlambdadist,lambdamum,lambdamum0)
targetformodel = np.pad(targetformodel, (0, max_length - len(targetformodel)), mode='constant') 
targetformodel=targetformodel.reshape(1, -1)
#predtree=np.append(predarrayabc[indextest],[0,0,0,0,0])
#predclass=np.append(predtestlecturesabc[indextest],[0,0,0,0,0])
predtree=np.array([modelDT1.predict(targetformodel),modelDT3.predict(targetformodel),modelDT2.predict(targetformodel)])
predtree=np.append(predtree,[0,0,0,0,0])
predclass=toabc(1/2*lambdamum0/DeltaL,ngFSR(targetforplot, dlambdadist, lambdamum, lambdamum0, DeltaL)[2],ngFSR(targetforplot, dlambdadist, lambdamum, lambdamum0, DeltaL)[3],lambdamum0)
predclass=np.append(predclass,[0,0,0,0,0])

modenumber=round(naben*DeltaL-1/2)
shiftna=(2*modenumber+1)/(2*DeltaL)
axes[1].plot(lambdamumHR,(lambdamum0+dlambdamumHR)*neffoverlambdaabc(naben,nbben,ncben,dlambdamumHR),color='b',linewidth=0.8) ## Blue is true
axes[1].plot(lambdamumHR,(lambdamum0+dlambdamumHR)*neffoverlambdaabc(predtree[0]+shiftna,predtree[1],predtree[2],dlambdamumHR),color='r',linewidth=0.8) ## Red is tree
axes[1].plot(lambdamumHR,(lambdamum0+dlambdamumHR)*neffoverlambdaabc(predclass[0]+shiftna,predclass[1],predclass[2],dlambdamumHR)
,color='g',linestyle='--',linewidth=0.6)  ## Green is class
axes[1].set_xticklabels([])
axes[1].set_ylabel('$n_{\mathrm{eff}}$')


axes[2].plot(lambdamumHR,TransferFdblist(dlambdamumHR, DeltaL,np.append(predtrue[0:3],[0,0,0,0,0])),color='blue',linewidth=0.8) ## Blue is true
axes[2].plot(lambdamumHR,TransferFdblist(dlambdamumHR, DeltaL,predtree),color='r',linewidth=0.8) ## Red is tree
axes[2].plot(lambdamumHR,TransferFdblist(dlambdamumHR, DeltaL,predclass),color='g',linestyle='--',linewidth=0.6) ## Green is class
axes[2].set_ylabel('$I_o/I_i$')
axes[2].set_xlabel('$\lambda [\mu\mathrm{m}]$')



fig.savefig('/Users/danielegana/Dropbox (Personal)/Si Photonics/report/figures/reconstruction.pdf', bbox_inches="tight")



##%%

##################################################################
##################################################################

#############
######
#%% Three parameter fit. It doesn't work for the phase.
#%%
X_trainall, X_testall, y_trainall, y_testall = train_test_split(features, labels, test_size=0.4, random_state=30)
y_testall[:,0]=(2*np.pi*y_testall[:,0]*DeltaL % (math.tau/2))/(2*np.pi*DeltaL) #Make sure na is predicted modulo Pi
#%%
modelDTall=ensemble.RandomForestRegressor(n_jobs=-1)
modelDTall.fit(X_trainall,y_trainall)
#%%

y_predall = modelDTall.predict(X_testall)
y_predtrainall = modelDTall.predict(X_trainall)

#%%
frac_error=np.divide(y_predtrainall,y_trainall)
mse = mean_squared_error(y_predtrainall,y_trainall)
mae = np.sqrt(mean_squared_error(y_predtrainall,y_trainall))
print("MFRE per feature",np.mean(frac_error, axis=0))
print("MSE DT: ",mse)
print("MAE DT: ",mae)

#%%
y_predall[:,0]=(2*np.pi*y_predall[:,0]*DeltaL % (math.tau/2))/(2*np.pi*DeltaL) #Make sure na is predicted modulo Pi
frac_error=np.divide(y_predall,y_testall)
mse = mean_squared_error(y_testall, y_predall)
mae = np.sqrt(mean_squared_error(y_testall, y_predall))
print("MFRE per feature",np.mean(frac_error, axis=0))
print("MSE DT: ",mse)
print("MAE DT: ",mae)
#%%


######## Test with Neural Networks ###########
#####################################################################
#%% getDataset class. label and feature dimensions are set here and in mynet
class getDataset(Dataset):
    def __init__(self, labels, feature_functions,dlambdaarray,DeltaLmum):
        self.labels = labels
        self.feature_functions=feature_functions
        self.dlambdaarray=dlambdaarray
        self.DeltaLmum=DeltaLmum
    #    self.lambda0mum=lambda0mum

    def __len__(self):
        # Number of instances in the dataset
        return len(self.labels)

    def __getitem__(self, idx):
        # Select the label row at the given index
        labels = self.labels[idx]

        # Apply each feature function to the labels
        features = self.feature_functions(self.dlambdaarray,self.DeltaLmum,labels)

        # Concatenate the features if necessary

        # Return a tuple of (label, features)
       # labelsnorm=np.column_stack((labels[0]/n0max,labels1]/n1max,labels[2]/n2max))
        labels=torch.tensor(labels[2], dtype=torch.float32).to(device)

        features=torch.tensor(features, dtype=torch.float32).to(device)
        return features,labels


#%% Define Neural Network

class mynet(nn.Module):    
    def __init__(self,featuredimensions,labeldimensions):
        super(mynet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(featuredimensions, 200),       
          #  nn.BatchNorm1d(64),        
            nn.ReLU(),                 
            nn.Linear(200, 100),         # Hidden layer
           # nn.BatchNorm1d(32),        
            nn.ReLU(),          
            nn.Linear(100, 50),         # Hidden layer
            nn.ReLU(),     
            nn.Linear(50, 10),         # Hidden layer
            nn.ReLU(),       
            nn.Linear(10, labeldimensions)           # Output layer
        )
    
    def forward(self, x):
        return self.model(x)
    

#Defines the function to train the model
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        #optimizer.zero_grad() zeroes out the gradient after one pass. this is to 
        #avoid accumulating gradients, which is the standard behavior
        optimizer.zero_grad()

        # Print loss every 100 batches
        if batch % 100 == 0:
            loss, current = loss.item(), batch*batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}]")

# %%
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print("Avg loss: ",test_loss)

#%% Neural network
#getDataset(labels, feature_functions,lambdaarray,DeltaLmum,lambda0mum)
#labelsforNN=[np.concatenate((row,np.array([labelben[3],labelben[4],labelben[5],labelben[6],labelben[7]]))) for row in labels]
#labelsforNN=np.array(labelsforNN)
#%%
MZIdata=getDataset(labels,TransferFdblist,dlambdamum,DeltaL)
train_dataset = Subset(MZIdata, range(0,train_size))
test_dataset = Subset(MZIdata, range(train_size,len(MZIdata)))

#%%featuredimensions,labeldimensions
model = mynet(featuredimensions=numpointslambda,labeldimensions=1).to(device)
batch_size=10
epochs=20
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), learningrate)    
#%%
learningrate=1e-6

#%%

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)    
    print("Done!")
#%%
test(test_loader, model, loss_fn)
#features=np.array([TransferFdblist(numpointslambda, 1.31, DeltaL,row) for row in labels])
#featuresben=np.array([TransferFdblist(numpointslambda, 1.31, DeltaL,labelben)])
#%%
features=np.array([TransferFdblist(lambdamum, 1.31, DeltaL,np.concatenate((row,np.array([labelben[3],labelben[4],labelben[5],labelben[6],labelben[7]])))) for row in labels])
featuresben=np.array([TransferFdblist(lambdamum, 1.31, DeltaL,labelben)])
#%%
model.eval()


######################################################################

