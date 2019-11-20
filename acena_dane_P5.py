#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:20:39 2019

@author: Dane Acena

Project 2: Regression

references: https://github.com/pickus91/Polynomial-Regression-From-Scratch/blob/master/polynomial_regression.py
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from statistics import mean, median
from pylab import rcParams

names = ['Clump Thickness',
         'Uniformity of Cell Size',
         'Uniformity of Cell Shape',
         'Marginal Adhesion',
         'Single Epithelial Cell Size',
         'Bare Nuclei','Bland Chromatin',
         'Normal Nucleoli',
         'Mitoses', 
         'Class']
    
def plotCorrelation(data):
    data.columns = names
    rcParams['figure.figsize'] = 15,20
    fig = plt.figure()
    sns.heatmap(data.corr(), annot=True, fmt=".2f")
    plt.show()

def plotDensity(data):
    data.columns = names
    rcParams['figure.figsize'] = 15, 20
    
    benign = data[data['Class'] == 2]
    malignant = data[data['Class'] == 4]
    
    fig, axs = plt.subplots(9,1)
    fig.suptitle('Features densities for different outcomes 2/4')
    fig.subplots_adjust(left = 0.25, right =0.9, bottom = 0.1, top = 0.95,
                        wspace = 0.2, hspace = 0.9)
    
    for columnName in names[:-1]:
        ax = axs[names.index(columnName)]
        benign[columnName].plot(kind='density', ax=ax, subplots=True,
                                sharex=False, color='red', legend=True,
                                label=columnName + 'for Outcome = 2')
        malignant[columnName].plot(kind='density', ax=ax, subplots=True,
                                sharex=False, color='green', legend=True,
                                label=columnName + 'for Outcome = 4')
        ax.set_xlabel(columnName + ' values')
        ax.set_title(columnName + ' density')
        ax.grid('on')
    
    plt.show()
    
#def pairplot(data):
                        
"""
 Loads data from input and spliting it to two(70-30): Training Set(210) and Test 
 Set(90) Save the Training Set and Test Set as txt files. 
 
 INPUT: filename from user input, split = a float that denotes the % of split
 OUTPUT: trainingSet and testSet dataframes and saved into text files
"""
def loadDataSet(filename, split):
    # uses pandas library to read a tab-delimited text file from user input
    # headers are removed
    data = pd.read_csv(filename, delimiter=',',header=None, usecols=range(1,11),names=names)
    
    data = cleanData(data)
    
    normalizedData = normalizeData(data, ['Clump Thickness',
                                          'Uniformity of Cell Size',
                                          'Uniformity of Cell Shape',
                                          'Marginal Adhesion',
                                          'Single Epithelial Cell Size',
                                          'Bare Nuclei','Bland Chromatin',
                                          'Normal Nucleoli',
                                          'Mitoses'])#, range(1,10)
    
    # Randomizes gpaDataSet and puts it back to gpaDataSet
    normalizedRandomData = normalizedData.sample(frac=1).reset_index(drop=True)
#    print(normalizedRandomData)
    
    # gpaDataSet is split depending on input a 70 on split gives 70-30.
    # trainingSet gets 70% and testSet get 30%
    trainingSet, testSet = splitTrainTest(normalizedRandomData,split)
    
    return(data, normalizedRandomData)
    
def cleanData(data):
    data['Bare Nuclei'] = pd.to_numeric(data['Bare Nuclei'].replace('?', data['Bare Nuclei'].replace(['?'], [None])))
    colMeans = data['Bare Nuclei'].mean()
    data['Bare Nuclei'] = data['Bare Nuclei'].fillna(int(colMeans))
    for col in data['Bare Nuclei']:
#        data[col] = data[col].astype(int)
        col = int(col)
    return data
 
def normalizeData(data, normalizeColumns = None):
    if normalizeColumns == None:
        data  = (data - data.mean()) / data.std()
    else:
        for col in normalizeColumns:
            data[col] = (data[col]-data[col].mean())/data[col].std()
    return data

def splitTrainTest(data,split):
    trainingSet, testSet = np.split(data, [int(split*len(data))])
    
    # saves trainingSet and testSet into two text files
    np.savetxt("AcenaTrainingSet.txt", trainingSet, fmt='%g', delimiter=',')
    np.savetxt("AcenaTestSet.txt", testSet, fmt='%g', delimiter=',')
    
    return (trainingSet, testSet)
            
#def predict(row, coefficients):
#    yhat = coefficients.iloc[0,0]
#    for i in range(len(row)-1):
#        yhat += coefficients.iloc[i+1,0] * row[i]
#    return yhat
    
def sigmoid(cThickness, cSize, mAd, sSize, bNuc,bChroma, nNucleoli, mitoses, theta):
    h = (1/(1+(np.exp(-1*(theta[0] + 
                         (theta[1] * cThickness) + 
                         (theta[2] * cSize) + 
                         (theta[3] * mAd) + 
                         (theta[4] * sSize) + 
                         (theta[5] * bNuc) + 
                         (theta[6] * bChroma) + 
                         (theta[7] * nNucleoli) + 
                         (theta[8] * mitoses)))))) 
    return h 

def hypothesisFunc(cThickness, cSize, mAd, sSize, bNuc,bChroma, nNucleoli, mitoses, theta):
    return (theta[0] + 
            (theta[1] * cThickness) + 
            (theta[2] * cSize) + 
            (theta[3] * mAd) + 
            (theta[4] * sSize) + 
            (theta[5] * bNuc) + 
            (theta[6] * bChroma) + 
            (theta[7] * nNucleoli) + 
            (theta[8] * mitoses)) 
 
    
def iterationsBestW(iterations, data, theta, alpha):
    costs = []
    iteration = []
    for x in range(iterations):
        theta = updateTheta(data, theta, alpha)
        cost = getJCostByTheta(data, theta)
        costs.append(cost)
        iteration.append(x)
        
    jCostPlot(iteration, costs)
    
    return theta, costs[-1]

def updateTheta(data, theta, alpha):
    thetaUpdated = [None] * len(theta)
    for w in range(len(theta)):
        
        m = len(data)
        
        prevTheta = theta[w]
        
        tempTheta = 0
        for i in range(len(data)): 
            cThickness = data['Clump Thickness'].iloc[i]
            cSize  = data['Uniformity of Cell Size'].iloc[i]
            mAd  = data['Marginal Adhesion'].iloc[i]
            sSize  = data['Single Epithelial Cell Size'].iloc[i]
            bNuc  = data['Bare Nuclei'].iloc[i]
            bChroma  = data['Bland Chromatin'].iloc[i]
            nNucleoli  = data['Normal Nucleoli'].iloc[i]
            mitoses  = data['Mitoses'].iloc[i]
            class_bm     = data['Class'].iloc[i]
            
            newX = getNewX(w, cThickness, cSize, mAd, sSize, bNuc,bChroma, nNucleoli, mitoses)
#            print("theta: ", theta)
            tempTheta += (hypothesisFunc(cThickness, cSize, mAd, sSize, bNuc,bChroma, nNucleoli, mitoses, theta) - class_bm) * newX
        
        newTheta =  prevTheta - alpha * (1/m) * tempTheta

        thetaUpdated[w] = newTheta
#    print(thetaUpdated)
    return thetaUpdated 

def getNewX(theta, cThickness, cSize, mAd, sSize, bNuc,bChroma, nNucleoli, mitoses):
    if theta == 0:
        return 1
    elif theta == 1:
        return float(cThickness)
    elif theta == 2:
        return float(cSize)
    elif theta == 3:
        return float(mAd)
    elif theta == 4:
        return float(sSize)
    elif theta == 5:
        return float(bNuc)
    elif theta == 6:
        return float(bChroma)
    elif theta == 7:
        return float(nNucleoli)
    elif theta == 8:
        return float(mitoses)

def getJCostByTheta(data, theta):
    cost = 0
    m = len(data)
    for i in range(len(data)):
        cThickness = data['Clump Thickness'].iloc[i]
        cSize  = data['Uniformity of Cell Size'].iloc[i]
        mAd  = data['Marginal Adhesion'].iloc[i]
        sSize  = data['Single Epithelial Cell Size'].iloc[i]
        bNuc  = data['Bare Nuclei'].iloc[i]
        bChroma  = data['Bland Chromatin'].iloc[i]
        nNucleoli  = data['Normal Nucleoli'].iloc[i]
        mitoses  = data['Mitoses'].iloc[i]
        class_bm     = data['Class'].iloc[i]

        cost += ((hypothesisFunc(cThickness, cSize, mAd, sSize, bNuc,bChroma, nNucleoli, mitoses, theta) - class_bm) ** 2)
    
#    print(theta)
#    print("J: " + repr((1/(2*m) * cost)))
  
    return ((1/(2 * m))*cost)

def jCostPlot(iteration, costs):
    plt.scatter(iteration, costs, marker = '.')
    plt.xlabel("Iterations")
    plt.ylabel("Cost(J)")
    plt.title("Cost(J) over Iteration")
    plt.savefig("acenaCostPlot.png")
    plt.show()
    return

def getPredictions(data, theta):
    threshold = 2.5
    predictions = []
    for i in range(len(data)):
        cThickness = data['Clump Thickness'].iloc[i]
        cSize  = data['Uniformity of Cell Size'].iloc[i]
        mAd  = data['Marginal Adhesion'].iloc[i]
        sSize  = data['Single Epithelial Cell Size'].iloc[i]
        bNuc  = data['Bare Nuclei'].iloc[i]
        bChroma  = data['Bland Chromatin'].iloc[i]
        nNucleoli  = data['Normal Nucleoli'].iloc[i]
        mitoses  = data['Mitoses'].iloc[i]
        gpa     = hypothesisFunc(cThickness, cSize, mAd, sSize, bNuc,bChroma, nNucleoli, mitoses, theta)
        if gpa <= threshold:
            predictions.append(2)
        else:
            predictions.append(4)
        
    actualGPA    = pd.DataFrame({'Actual GPA':data['Class']})
    predictionGPA = pd.DataFrame({'Predicted GPA':predictions})
    comparisonGPA = pd.concat([actualGPA, predictionGPA],axis=1)
    return comparisonGPA

def getError(comparisonGPA):
    errors = []
    
    for x in range(len(comparisonGPA)):
        actual     = comparisonGPA['Actual GPA'].iloc[x]
        prediction = comparisonGPA['Predicted GPA'].iloc[x]
        error      = abs(prediction - actual)/actual
        errors.append(error)
#        print("Expected: %.2f, Predicted: %.2f, Error:, %.2f" % (actual, 
#                                                                 prediction, 
#                                                                 error))
    return mean(errors), median(errors)

def values(theta, testJ, testSet):
    
    predictionVActual = getPredictions(testSet, theta)
    mean_error, median_error = getError(predictionVActual)
    
    print("Values: ")
    for x in range(len(theta)):
        print("w"+repr(x) + ": " + repr(theta[x].round(4)))
    
    print("J: " + repr(testJ.round(4)))
#    print("Mean error on test set: " + repr(mean_error.round(4)))
#    print("Median error on test set: " + repr(median_error.round(4)))
    
def meansAndStd(data):
    dataColMean = {}
    dataColStd = {}
    
    for col in data:
        dataColMean[col] = data[col].mean()
        dataColStd[col] = data[col].std()
    return dataColMean, dataColStd
    
    
def userInputs(theta, dataColMean, dataColStd):
    print(dataColMean)
    print(dataColStd)
    while True:
        studyMinutes = float(input("Minutes spent studying each week: "))
        beerOunces   = float(input("Ounces of beer each week: "))
        if studyMinutes == 0 and beerOunces == 0:
            break
        elif studyMinutes < 0 and beerOunces < 0:
            print("Those value needs to be positive.")
        else:
            studyNorm = (studyMinutes - dataColMean['studyMinutes'])/dataColMean['studyMinutes']
            beerNorm = (beerOunces - dataColStd['ozBeer'])/dataColStd['ozBeer']
            
            print(studyNorm)
            print(beerNorm)
        
            prediction = hypothesisFunc(studyNorm, beerNorm, theta)
            
            print("With those value, GPA is predicted to be: " + str(prediction.round(2)))
            
def test(data, theta):
    print("data",data)
    meanData, stdData = meansAndStd(data) 
    userInputs(theta, meanData, stdData)
    
def divider():
    print("==================================================================")

def confusionMatrix(y,outcome):
    f1_score = k = tp = fp = fn = tn = 0
    for i in range(len(outcome)):
        if outcome[i] == y[i]:
            k = k + 1
        if outcome[i] == y[i] == 2:
            tp = tp + 4
        if outcome[i] == 2 and y[i] == 4:
            fp = fp + 1
        if outcome[i] == 4 and y[i] == 2:
            fn = fn + 1
        if (outcome[i] == y[i] == 4):
            tn = tn + 1
            
    accuracy = k/len(y)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1_score = (2 * precision * recall)/(precision + recall)
            
    cm = np.array([[tp, fn], [fp, tn]])
    
    print("True Positive: ", tp)
    print("True Negative: ", tn)
    print("False Positive: ", fp)
    print("False Negative: ", fn)
    print("Recall: ", recall)
    print("precision: ", precision)
    print("Accuracy: ", accuracy)
    print("F1 Score: ", f1_score)
    
        # Un-Normalized Confusion Matrix...
    plotConfusionMatrix(cm, classes=[0,1])

    
def plotConfusionMatrix(cm, classes):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cmap = plt.cm.Blues
    print('Confusion matrix, without normalization')
#    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
        
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def main():
    print("""
         +-+-+-+-+-+-+ +-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+-+
         |B|r|e|a|s|t| |C|a|n|c|e|r| |P|r|e|d|i|c|t|i|o|n|
         +-+-+-+-+-+-+ +-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+-+
      """)
    filename = "breast-cancer-wisconsin.data"
    split = 0.80
    alpha = .1
    theta = [1,1,1,1,1,1,1,1,1]
    jVal = 0.0
    iterations = 60
    
    dataset, normalizedData = loadDataSet(filename,split)
    
    y_target = normalizedData['Class']
    
    normalizedData['target'] = normalizedData['Class'].map({2:'B',4:'M'}) 
    
#    print(y_target)    
    
#    plotCorrelation(normalizedData.iloc[:,0:10])
#    
#    plotDensity(dataset)
    
#    print(dataSet)
    
    trainSet = pd.read_csv("AcenaTrainingSet.txt", delimiter=',', header=None,
                           names=names)
    testSet = pd.read_csv("AcenaTestSet.txt", delimiter=',', header=None,
                           names=names)
    
    theta, jVal = iterationsBestW(iterations, trainSet, theta, alpha)
##    print(theta)
##    print (jVal)
#    
    divider()
    
    comparisonGPA = getPredictions(testSet, theta)
    confusionMatrix(comparisonGPA['Actual GPA'], comparisonGPA['Predicted GPA'])
    values(theta,jVal,testSet)
    print(jVal)
    print(theta)
#    while True:
#        print("""
#            ==============================================================
#            | (1) If you want to see the plot for the original data set  |
#            |  select use 'plot'                                         |
#            |                                                            |
#            | (2) To find out theta (weights) value and cost by alpha and|
#            |  iteration value input use 'train'.                        |
#            |                                                            |
#            | (3) To see the final theta values, j value use on test set |
#            |  and mean and median error on the test set use 'validate'. |
#            |                                                            |
#            | (4) To predict a the student's GPA by input of minutes     |
#            |  spent studying and ounce of beer per week use 'test'.     |
#            |                                                            |
#            |   If you want to reset and change the dataset file         |
#            |   use 'reset'.                                             |
#            |                                                            |
#            |   If you want to terminate the program use 'end'           |
#            ==============================================================
#              """)
#        mode = str(input("What mode? : "))
#
#        if mode == 'plot':
#            plotData(filename)
#            continue
#
#        elif mode == 'train':
#            theta, jVal = iterationsBestW(iterations, trainSet, theta, alpha)
#            print(theta)
#            print (jVal)
#
#        elif mode == 'validate':
#            comparisonGPA = getPredictions(testSet, theta)
#            getError(comparisonGPA)
#            values(theta,jVal,testSet)
#            print(jVal)
#            print(theta)
#            # 50 iterations, alpha = 0.1:
#            # Mean error on test set: 0.5889
#            # Median error on test set: 0.0963
#            # [0.9237257215271041, 1.1703094346154619, 0.014198617048305016, 0.014488271778922516, 0.4011144294407487, 0.0881715678208272]
#
#            # 60 iterations, alpha = 0.1:
#            # Mean error on test set: 0.2067
#            # Median error on test set: 0.0663
#            # [0.9832787538655158, 1.1807664608037387, 0.009547613721306579, 0.015053800852169334, 0.37404043119987607, 0.07029230117825747]
#
#            # 70 iterations, alpha = 0.1:
#            # Mean error on test set: 0.2332
#            # Median error on test set: 0.068
#            # [1.0133336428793047, 1.169287085941109, 0.008250836881047312, 0.0005560147637625505, 0.35863062356926984, 0.05559701937387916]
#
#            # 80 iterations, alpha = 0.1:
#            # Mean error on test set: 0.2568
#            # Median error on test set: 0.0445
#            # [1.054051762379865, 1.1731813261357207, 0.007222586662875618, 0.0015769751611844744, 0.34311208799194287, 0.04340417380980185]
#        elif mode == 'test':
#            test(gpaDataSet, theta)
#
#        elif mode == 'end':
#            break
#
#        elif mode == 'reset':
#            main()
#
#        else:
#            print("Invalid response, please use one of the options below")
#            continue

        
main()
