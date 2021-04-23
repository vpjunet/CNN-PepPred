import os
import sys
import numpy as np
import numpy.matlib
import pandas as pd
import time
from model_initializer import CNNPepPred
from pathlib import Path
from Bio import SeqIO


def readFasta(file):
    fasta_sequences = SeqIO.parse(open(file),'fasta')
    seq = []
    seqName = []
    for fasta in fasta_sequences:
        nameCurr, seqCurr = fasta.id, str(fasta.seq)
        seq.append(seqCurr)
        seqName.append(nameCurr)
    return seq, seqName
    
def readTemplate(file):
    valueColName = "Input_Value"
    tmpl = pd.read_csv(file,index_col='Input')
    allele = tmpl[valueColName]["allele"]
    savePath = tmpl[valueColName]["savePath"]
    if savePath==savePath:
        savePath = Path(savePath)
    else:
        savePath = Path(os.getcwd())
    doTraining = tmpl[valueColName]["doTraining"]
    if doTraining!=doTraining:
        doTraining = False
    else:
        doTraining = int(doTraining)>0
    trainingDataPath = tmpl[valueColName]["trainingDataPath"]
    doLogoSeq = tmpl[valueColName]["doLogoSeq"]
    if doLogoSeq!=doLogoSeq:
        doLogoSeq = False
    else:
        doLogoSeq = int(doLogoSeq)>0
    doCV = tmpl[valueColName]["doCV"]
    if doCV!=doCV:
        doCV = False
    else:
        doCV = int(doCV)>0
    kFold = tmpl[valueColName]["kFold"]
    if kFold!=kFold:
        kFold = 5
    doApplyData = tmpl[valueColName]["doApplyData"]
    if doApplyData!=doApplyData:
        doApplyData = False
    else:
        doApplyData = int(doApplyData)>0
    trainedModelsFile = tmpl[valueColName]["trainedModelsFile"]
    if trainedModelsFile==trainedModelsFile:
        trainedModelsFile = Path(trainedModelsFile)
    else:
        trainedModelsFile = None
    applyDataPath = tmpl[valueColName]["applyDataPath"]
    epitopesLength = tmpl[valueColName]["epitopesLength"]
    if epitopesLength!=epitopesLength:
        epitopesLength = 15
    parametersFile = tmpl[valueColName]["parametersFile"]
    if parametersFile!=parametersFile:
        parametersFile = 'parameters.txt'
    saveClassObject = tmpl[valueColName]["saveClassObject"]
    if saveClassObject!=saveClassObject:
        saveClassObject = False
    else:
        saveClassObject = int(saveClassObject)>0
    if doTraining:
        trainingDataPath = Path(trainingDataPath)
        trainingDataAll = pd.read_csv(trainingDataPath)
        trainingData = trainingDataAll.iloc[:,0].to_list()
        trainingOutcome = trainingDataAll.iloc[:,1].to_numpy()
        if doCV:
            if trainingDataAll.shape[1]>2:
                cvPart = trainingDataAll.iloc[:,2].to_numpy(dtype=int)
                if np.isnan(cvPart).any():
                    cvPart = None
                    print('The given cross-validation partition contains some NaN values. A new partition will be generated using the default function.')
                    kFold = int(kFold)
                else:
                    kFold = np.max(cvPart)+1
            else:
                cvPart = None
                kFold = int(kFold)
        else:
            cvPart = None
    else:
        trainingData = None
        trainingOutcome = None
        doCV = False
        cvPart = None
    if doApplyData:
        applyDataPath = Path(applyDataPath)
        [applyData,applyDataName] = readFasta(applyDataPath)
        epitopesLength = int(epitopesLength)
    else:
        applyData = None
        applyDataName = None
    return allele,savePath,doTraining,trainingData,trainingOutcome,doLogoSeq,doCV,cvPart,kFold,doApplyData,trainedModelsFile,applyData,applyDataName,epitopesLength,parametersFile,saveClassObject
     
def main(tmplName):
    time_start = time.perf_counter()
    file = Path(tmplName)
    allele,savePath,doTraining,trainingData,trainingOutcome,doLogoSeq,doCV,cvPart,kFold,doApplyData,trainedModelsFile,applyData,applyDataName,epitopesLength,parametersFile,saveClassObject = readTemplate(file)
    modelCNN = CNNPepPred(allele,savePath,doTraining,trainingData,trainingOutcome,doLogoSeq,doCV,cvPart,kFold,doApplyData,trainedModelsFile,applyData,applyDataName,epitopesLength,parametersFile)
    print('Model initialized \n')
    if modelCNN.doTraining:
        sInt = modelCNN.aa2int(modelCNN.trainingData)
        modelCNN.seqLength(sInt,saveOutput=True)
        sInt = modelCNN.addEmptyPositions(sInt)
        IM = modelCNN.getImages(sInt)
        out = modelCNN.trainingOutcome
        if modelCNN.doCV:
            print('Doing: CV \n')
            modelCNN.crossValCNN(IM,out)
            modelCNN.getCVresults()
            print('CV: done \n')
        print('Doing: Training \n')
        modelCNN.trainCNN(IM,out,saveModel=True)
        print('Training: done \n')
        
    if modelCNN.doLogoSeq:
        print('Doing: LogoSeq \n')
        sR = modelCNN.generateRandomSeq()
        contributionScore,yhatR = modelCNN.feedForwardAndGetScore(sR)
        modelCNN.plotLogoSeq(contributionScore,yhatR)
        print('LogoSeq: Done \n')
        
    if modelCNN.doApplyData:
        print('Doing: Apply \n')
        sIntApply,sApplyName = modelCNN.seq2Lmer(modelCNN.aa2int(modelCNN.applyData),L=None,nameSeq=modelCNN.applyDataName,saveLmer = True)[0:2]
        sIntApply = modelCNN.addEmptyPositions(sIntApply)
        modelCNN.feedForwardAndGetScore(sIntApply,saveOutcome = True)
        modelCNN.getCoreBinder(modelCNN.int2aa(sIntApply),modelCNN.contributionScore,sApplyName,saveCoreBinders = True)
        modelCNN.printApplyOutcome()
        print('Apply: Done \n')
           
    time_elapsed = (time.perf_counter() - time_start)
    modelCNN.computationTime(time_elapsed)
    if saveClassObject:
        modelCNN.save_object()
    print('Finished \n')
    return modelCNN
    
if __name__=='__main__':
    tmplName = sys.argv
    if len(tmplName)==1:
        tmplName = 'template.txt'
    else:
        tmplName = tmplName[1]
    modelCNN = main(tmplName)