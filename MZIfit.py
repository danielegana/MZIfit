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

# These are the 7 param tuples. In addition to the BH params, we have a random seed for different runs.
n0= np.random.permutation(np.linspace(n0min, n0max, lendata))
n1= np.random.permutation(np.linspace(n1min, n1max, lendata))
n2= np.random.permutation(np.linspace(n2min, n2max, lendata))

off0 = np.random.permutation(np.linspace(off0min, off0max, lendata))
off1 = np.random.permutation(np.linspace(off1min, off1max, lendata))
off2 = np.random.permutation(np.linspace(off2min, off2max, lendata))
off3 = np.random.permutation(np.linspace(off3min, off3max, lendata))
off4 = np.random.permutation(np.linspace(off4min, off4max, lendata))
randomseed=np.random.randint(0,1e9,lendata)

#%%

def neff(n0,n1,n2,dlambdamum):
    return n0+n1*dlambdamum+n2*dlambdamum**2

#Label definitions
labels=np.column_stack((n0,n1,n2))
labels=np.array([row for row in labels if neff(row[0],row[1],row[2],bwdtmum) <= 1.7 and neff(row[0],row[1],row[2],bwdtmum) >= 1.6])
labelsnorm=np.column_stack((labels[:,0]/n0max,labels[:,1]/n1max,labels[:,2]/n2max))

labelben=np.array([1.67194,-0.324256,0.0678241,-5.507123133079745,-0.07110396303254388,-0.0024757350518301766,-0.00001588506651858012,-1.2176688398843063*10**(-7)])
labelbennorm=np.array([1.67194/n0max,-0.324256/n1max,0.0678241/n2max,-5.507123133079745/maxoffset,-0.07110396303254388/off1max,-0.0024757350518301766/off2max,-0.00001588506651858012/off3max,-1.2176688398843063*10**(-7)/off4max])


train_size = int(0.8 * labels.shape[0])
val_size = labels.shape[0] - train_size
# labels=np.column_stack((n0,n1,n2,off0,off1,off2,off3,off4))
# labelsnorm=np.column_stack((n0/n0max,n1/n1max,n2/n2max,off0/maxoffset,off1/off1max,off2/off2max,off3/off3max,off4/off4max))

#Lambda array definition
numpointslambda=501
lambdamummin=1.26
lambdamum0=1.31
lambdamummax=1.36
lambdamum=np.linspace(lambdamummin,lambdamummax,numpointslambda)
#normalized_grid = (lambdamum - lambdamum0) / (lambdamummax - lambdamum0)
#transformed_grid = np.array([np.sqrt(x) if x>0 else -np.sqrt(-x) for x in normalized_grid])
#lambdamum = lambdamu0 + (lambdamummax - lambdamum0) * transformed_grid

#%%
DeltaL=151.47

#%% Define Transfer Function

def TransferFdblist(lambdamum,lambda0mum,DeltaLmum,labelarray):
    dlambdamum=lambdamum-lambda0mum
    dlambdanm=dlambdamum*1.0e3
    neff=labelarray[0]+labelarray[1]*dlambdamum+labelarray[2]*dlambdamum**2
    #offset=labelarray[3]+labelarray[4]*dlambdanm+labelarray[5]*dlambdanm**2+labelarray[6]*dlambdanm**3+labelarray[7]*dlambdanm**4
    beta=neff*2*np.pi/(lambdamum)
    #TFdB=10*np.log10(0.5*(1+np.cos(beta*DeltaLmum)))+2*offset
    TFdB=10*np.log10(0.5*(1+np.cos(beta*DeltaLmum)))
    return TFdB

#%% getDataset class. label and feature dimensions are set here and in mynet
class getDataset(Dataset):
    def __init__(self, labels, feature_functions,lambdaarray,DeltaLmum,lambda0mum):
        self.labels = labels
        self.feature_functions=feature_functions
        self.lambdaarray=lambdaarray
        self.DeltaLmum=DeltaLmum
        self.lambda0mum=lambda0mum

    def __len__(self):
        # Number of instances in the dataset
        return len(self.labels)

    def __getitem__(self, idx):
        # Select the label row at the given index
        labels = self.labels[idx]

        # Apply each feature function to the labels
        features = self.feature_functions(self.lambdaarray,self.lambda0mum,self.DeltaLmum,labels)

        # Concatenate the features if necessary

        # Return a tuple of (label, features)
        labelsnorm=np.column_stack((labels[0]/n0max,labels1]/n1max,labels[2]/n2max))
        labels=torch.tensor(labels, dtype=torch.float64).to(device)

        features=torch.tensor(features, dtype=torch.float64).to(device)
        return features,labelsnorm


#%% Define Neural Network

class mynet(nn.Module):    
    def __init__(self,featuredimensions,labeldimensions):
        super(mynet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(featuredimensions, 64),       
            nn.BatchNorm1d(64),        
            nn.ReLU(),                 
            nn.Linear(64, 32),         # Hidden layer
            nn.BatchNorm1d(32),        
            nn.ReLU(),                
            nn.Linear(32, labeldimensions)           # Output layer
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
MZIdata=getDataset(labels,TransferFdblist,lambdamum,DeltaL,lambdamum0)
train_dataset = Subset(MZIdata, range(0,train_size))
test_dataset = Subset(MZIdata, range(train_size,len(MZIdata)))

#%%featuredimensions,labeldimensions
model = mynet(featuredimensions=numpointslambda,labeldimensions=3).to(device)
batch_size=100
epochs=5
learningrate=1e-3
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), learningrate)    
#%%
learningrate=1e-4

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



#%% Decision Trees
# 3-parameter fit
features=np.array([TransferFdblist(lambdamum, 1.31, DeltaL,np.concatenate((row,np.array([labelben[3],labelben[4],labelben[5],labelben[6],labelben[7]])))) for row in labels])
featuresben=np.array([TransferFdblist(lambdamum, 1.31, DeltaL,labelben)])
X_train, X_test, y_train, y_test = train_test_split(features, labelsnorm, test_size=0.2, random_state=42)
#%%
modelDT=ensemble.RandomForestRegressor(n_jobs=-1,min_samples_leaf=2,max_depth=30,n_estimators=100)

#modelDT=tree.DecisionTreeRegressor(max_depth=1000)
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

#Decision trees, one parameter at a time
# First-parameter fit

lambdamumleft=lambdamum[int(numpointslambda*0.4)+1:int(numpointslambda*0.5)+1]#lambdamum[0:int(numpointslambda*0.3)]
lambdamumright=lambdamum[int(numpointslambda*0.5):int(numpointslambda*0.6)]
lambdamumright=lambdamumright[::-1]

#%%
targetfunction=[TransferFdblist(lambdamumleft, 1.31, DeltaL,np.concatenate((row,np.array([labelben[3],labelben[4],labelben[5],labelben[6],labelben[7]]))))+TransferFdblist(lambdamumright, 1.31, DeltaL,np.concatenate((row,np.array([labelben[3],labelben[4],labelben[5],labelben[6],labelben[7]])))) for row in labels]
features=np.array(targetfunction)
X_train, X_test, y_train, y_test = train_test_split(features, labels[:,0], test_size=0.3, random_state=30)
#%%

# Train the model

#modelDT1=ensemble.RandomForestRegressor(n_jobs=-1)

modelDT=tree.DecisionTreeRegressor()
modelDT1.fit(X_train,y_train)

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
frac_error=np.divide(y_pred,y_test)
mse = mean_squared_error(y_test, y_pred)
mae = np.sqrt(mean_squared_error(y_test, y_pred))
print("MFRE per feature",np.mean(frac_error, axis=0))
print("MSE DT: ",mse)
print("MAE DT: ",mae)
#%%

##THERE MUST BE AN ERROR IN THE CODE, I CAN'T BRING THE ERROR DOWN FROM 0.021 EXACTLY
ypredtotal=modelDT1.predict(features)
labelspred=np.column_stack((ypredtotal, np.zeros((len(ypredtotal), 2))))
#%%
testtransfer=TransferFdblist(lambdamum, 1.31, DeltaL,np.concatenate((labels[130],np.array([labelben[3],labelben[4],labelben[5],labelben[6],labelben[7]]))))
testtransfer2=TransferFdblist(lambdamum, 1.31, DeltaL,np.concatenate((labelspred[130],np.array([labelben[3],labelben[4],labelben[5],labelben[6],labelben[7]]))))

#%%
featurestrainn0pred=np.array([TransferFdblist(lambdamumleft, 1.31, DeltaL,np.concatenate((row,np.array([labelben[3],labelben[4],labelben[5],labelben[6],labelben[7]]))))+TransferFdblist(lambdamumright, 1.31, DeltaL,np.concatenate((row,np.array([labelben[3],labelben[4],labelben[5],labelben[6],labelben[7]])))) for row in labelspred])
differentialfeaturesn0=features-featurestrainn0pred

#%%
#Second parameter fit

#%%
lambdamumleft=lambdamum[0:int(numpointslambda*0.5)-1]#lambdamum[0:int(numpointslambda*0.3)]
lambdamumright=lambdamum[int(numpointslambda*0.5)+1:]
lambdamumright=lambdamumright[::-1]
#%%
targetfunction=[TransferFdblist(lambdamumleft, 1.31, DeltaL,np.concatenate((row,np.array([labelben[3],labelben[4],labelben[5],labelben[6],labelben[7]]))))-TransferFdblist(lambdamumright, 1.31, DeltaL,np.concatenate((row,np.array([labelben[3],labelben[4],labelben[5],labelben[6],labelben[7]])))) for row in labels]

features=np.array(targetfunction)
X_train, X_test, y_train, y_test = train_test_split(features, labels[:,1], test_size=0.2, random_state=42)
#%%
modelDT2=ensemble.RandomForestRegressor(n_jobs=-1,min_samples_leaf=4)

#modelDT=tree.DecisionTreeRegressor(max_depth=1000)
modelDT2.fit(X_train,y_train)

y_pred = modelDT2.predict(X_test)
y_predtrain = modelDT2.predict(X_train)

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
#Third parameter fit

#%%
lambdamumleft=lambdamum[0:int(numpointslambda*0.5)-1]#lambdamum[0:int(numpointslambda*0.3)]
lambdamumright=lambdamum[int(numpointslambda*0.5)+1:]
lambdamumright=lambdamumright[::-1]
#%%
zeroedtarget=

targetfunction=[TransferFdblist(lambdamumleft, 1.31, DeltaL,np.concatenate((row,np.array([labelben[3],labelben[4],labelben[5],labelben[6],labelben[7]]))))-TransferFdblist(lambdamumright, 1.31, DeltaL,np.concatenate((row,np.array([labelben[3],labelben[4],labelben[5],labelben[6],labelben[7]])))) for row in labels]

features=np.array(targetfunction)
X_train, X_test, y_train, y_test = train_test_split(features, labels[:,1], test_size=0.2, random_state=42)
#%%
modelDT2=ensemble.RandomForestRegressor(n_jobs=-1,min_samples_leaf=4)

#modelDT=tree.DecisionTreeRegressor(max_depth=1000)
modelDT2.fit(X_train,y_train)

y_pred = modelDT2.predict(X_test)
y_predtrain = modelDT2.predict(X_train)

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
##

def TransferFdb(lambdamum,lambda0mum,DeltaLmum,n0,n1,n2,off0,off1,off2,off3,off4):
    dlambdamum=lambdamum-lambda0mum
    dlambdanm=dlambdamum*1.0e3
    neff=n0+n1*dlambdamum+n2*dlambdamum**2
    offset=off0+off1*dlambdanm+off2*dlambdanm**2+off3*dlambdanm**3+off4*dlambdanm**4
    beta=neff*2*np.pi/(lambdamum)
    TFdB=10*np.log10(0.5*(1+np.cos(beta*DeltaLmum)))+2*offset
    return TFdB
    lambdamummin=1.26
    lambdamumax=1.36
    lambdamum=np.linspace(lambdamummin,lambdamumax,npoints)
    lambdamu0=1.31
    #normalized_grid = (lambdamum - lambdamu0) / (lambdamumax - lambdamu0)
    #transformed_grid = np.array([np.sqrt(x) if x>0 else -np.sqrt(-x) for x in normalized_grid])
    #lambdamum = lambdamu0 + (lambdamumax - lambdamu0) * transformed_grid
    dlambdamum=lambdamum-lambda0mum
    dlambdanm=dlambdamum*1.0e3
    neff=labelarray[0]+labelarray[1]*dlambdamum+labelarray[2]*dlambdamum**2
    offset=labelarray[3]+labelarray[4]*dlambdanm+labelarray[5]*dlambdanm**2+labelarray[6]*dlambdanm**3+labelarray[7]*dlambdanm**4
    beta=neff*2*np.pi/(lambdamum)
    #TFdB=10*np.log10(0.5*(1+np.cos(beta*DeltaLmum)))+2*offset
    TFdB=10*np.log10(0.5*(1+np.cos(beta*DeltaLmum)))
    return TFdB