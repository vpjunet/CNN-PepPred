# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:01:12 2020

@author: valentin.junet
"""
import os
import numpy as np
import numpy.matlib
from pathlib import Path
from Bio import SeqIO
import random

def generateRandomNonBinders(fastaSeqLoc,seqL=None,seq=None,prop=1,N=None,maxFiles=None):
    fastaSeqLoc = Path(fastaSeqLoc)
    if seqL is None and seq is None:
        if N is None:
            N = 2000
            N = np.int32(np.round(prop*N))
        seqL = 15*np.ones((N,),dtype=int)
    else:
        if seqL is None:
            seqL = np.zeros((len(seq),1),dtype=int)
            for i,s in enumerate(seq):
                seqL[i] = len(s)
        if N is None:
            N = seqL.shape[0]
        seqL = seqL.reshape((-1,))
        N = prop*N
        maxL = np.max(seqL)
        seqLhist = np.histogram(seqL,range(1,maxL+2))[0]
        a = N*seqLhist/np.sum(seqLhist)
        nbRandSeqPerLength = np.array([np.int32(np.round(a[i])) for i in range(0,len(a))])
        zeroL = np.where(nbRandSeqPerLength==0.)
        L = np.arange(0,maxL)+1
        nbRandSeqPerLength = np.delete(nbRandSeqPerLength,zeroL)
        L = np.delete(L,zeroL)
        seqL = np.repeat(L,nbRandSeqPerLength)
        N = seqL.shape[0]
    
    fileSeq = os.listdir(fastaSeqLoc)
    nbAllFiles = len(fileSeq)
    if maxFiles is None:
        maxFiles = nbAllFiles
    nbFiles = np.min((nbAllFiles,maxFiles))
    I = random.sample(range(0,nbAllFiles),nbFiles)
    seqInFile = []
    for i in I:
        seqInFilecurr = []
        fasta_sequences = SeqIO.parse(open(fastaSeqLoc / fileSeq[i]),'fasta')
        for fasta in fasta_sequences:
            seqInFilecurr.append(str(fasta.seq))
        seqInFile.append(seqInFilecurr)
    del fasta_sequences,seqInFilecurr
    seqNeg = []
    i = 0
    charValid = set('ARNDCQEGHILKMFPSTWYV')
    while i<N:
        l = seqL[i]
        indFile = random.randint(0,nbFiles-1)
        currFileSeq = seqInFile[indFile]
        nbSeq = len(currFileSeq)
        indSeq = random.randint(0,nbSeq-1)
        currSeq = currFileSeq[indSeq]
        currL = len(currSeq)
        if currL-l>0:
            indStart = random.randint(0,currL-l)
            indEnd = int(indStart+l)
            pept = currSeq[indStart:indEnd]
            if set(pept).issubset(charValid):
                seqNeg.append(pept)
                i += 1
    return seqNeg