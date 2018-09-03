import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
from math import *
df = pd.read_excel('knndatatrain.xlsx')
df.drop(['class'],1,inplace=True)
#df.drop(['output(ab)normal'],1,inplace=True)
df.drop(['age'],1,inplace=True)
df.drop(['sec'],1,inplace=True)
df.drop(['failprob'],1,inplace=True)
#df.drop(['quality'],1,inplace=True)



dftest = pd.read_excel('knndatatest.xlsx')
ytest = np.array(dftest['class'])
dftest.drop(['class'],1,inplace=True)
#dftest.drop(['output(ab)normal'],1,inplace=True)
dftest.drop(['age'],1,inplace=True)
dftest.drop(['sec'],1,inplace=True)
dftest.drop(['failprob'],1,inplace=True)
#dftest.drop(['quality'],1,inplace=True)

dfcv = pd.read_excel('knndatacv.xlsx')
y = np.array(dfcv['class'])
dfcv.drop(['class'],1,inplace=True)
#dfcv.drop(['output(ab)normal'],1,inplace=True)
dfcv.drop(['age'],1,inplace=True)
dfcv.drop(['sec'],1,inplace=True)
dfcv.drop(['failprob'],1,inplace=True)
#dfcv.drop(['quality'],1,inplace=True)


a = df.as_matrix()
test = dftest.as_matrix()
cv = dfcv.as_matrix()


testlijstlijst = []
sigmafeatures = []
trainlijstlijst = []
cvlijstlijst = []

num_instances, num_features = a.shape
num_instancestest, num_featurestest = test.shape
num_instancescv, num_featurescv = cv.shape

meanfeatures = []
num_instances = int(num_instances)
num_features = int(num_features)
num_instancestest = int(num_instancestest)
num_featurestest = int(num_featurestest)
num_featurescv = int(num_featurescv)
num_instancescv = int(num_instancescv)
################################################ defining the mean###########################
for i in range(num_features):
    mean = 0
    sum = 0
    for j in range(num_instances):
        sum = sum + a[j,i]
    mean = sum/num_instances
    meanfeatures.append(mean)
print(meanfeatures)
################################################# defining sigma##############################
for i in range(num_features):
    sigma = 0
    newsom = 0
    for j in range(1,num_instances):
        verschil = 0
        verschil = a[j,i] - meanfeatures[i]
        newsom = newsom + verschil*verschil
    sigma = sqrt(newsom/num_instances)
    sigmafeatures.append(sigma)
print(sigmafeatures)
################################################# probability destributions test set######################
for i in range(num_featurestest):
    getal = 0
    getal2 = 0
    testlijst = []
    for j in range(num_instancestest):
        getal = 1/(sqrt(2*pi)*sigmafeatures[i])
        getal2 = exp(-((test[j,i]-meanfeatures[i])**2)/(2*sigmafeatures[i]**2))
        product = getal*getal2
        testlijst.append(product)
    testlijstlijst.append(testlijst)
print(len(testlijstlijst))
################################################# probability destributions train set#####################

for i in range(num_features):
    getal = 0
    getal2 = 0
    trainlijst = []
    for j in range(num_instances):
        getal = 1/(sqrt(2*pi)*sigmafeatures[i])
        getal2 = exp(-((a[j,i]-meanfeatures[i])**2)/(2*sigmafeatures[i]**2))
        product = getal*getal2
        trainlijst.append(product)
    trainlijstlijst.append(trainlijst)
print(len(trainlijstlijst))
################################################# probability destributions cv set#####################
for i in range(num_featurescv):
    getal = 0
    getal2 = 0
    cvlijst = []
    for j in range(num_instancescv):
        getal = 1/(sqrt(2*pi)*sigmafeatures[i])
        getal2 = exp(-((cv[j,i]-meanfeatures[i])**2)/(2*sigmafeatures[i]**2))
        product = getal*getal2
        cvlijst.append(product)
    cvlijstlijst.append(cvlijst)
print(len(cvlijstlijst))
################################################ product maken trainset ####################################

producttrainlijst = []
kleinstepuntvoorlopig = 9999999
for i in range(num_instances):
    product = 1
    for j in range(len(trainlijstlijst)):
        product = product*trainlijstlijst[j][i]
    if product < kleinstepuntvoorlopig:
        kleinstepuntvoorlopig = product
        kleinstelement = i
        kleinstelement2 = j
    producttrainlijst.append(product)


kleinstepunttrain = min(producttrainlijst)
print('kleinste punt van de train set')
print(kleinstepunttrain)

################################################ product maken testset ####################################
producttestlijst = []
for i in range(num_instancestest):
    product = 1
    for j in range(len(testlijstlijst)):
        product = product*testlijstlijst[j][i]
    producttestlijst.append(product)

kleinstepuntest= min(producttestlijst)
print('kleinste punt van de test set')
print(kleinstepuntest)

################################################ product maken cvset ####################################
print('kleinste punt van de cv set')
productcvlijst = []
for i in range(num_instancescv):
    product = 1
    for j in range(len(cvlijstlijst)):
        product = product*cvlijstlijst[j][i]
    productcvlijst.append(product)

kleinstepuntcv= min(productcvlijst)
print(kleinstepuntcv)




################################################ flag or not to flag ####################################
klasse = []
for i in range (len(producttestlijst)):
    if producttestlijst[i] < kleinstepunttrain:
        klasse.append(1)
        anomaly = i 
        print(anomaly)
    else:
        klasse.append(0)
################################################ accuracy##############################################
accuraat = 0
accuraatheid = 0
for i in range(len(producttestlijst)):
    if klasse[i] == ytest[i]:
        accuraat = accuraat+1
accuraatheid = accuraat/len(producttestlijst)
print(accuraatheid)

