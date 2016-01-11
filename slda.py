# -*- coding: utf-8 -*-

import numpy as np
import scipy.special
import time, cPickle
from scipy import optimize

class sLDA:
    # Variational implimentation of smoothed LDA

    def __init__(self):
        # Initialize class 
        pass

    def setData(self,data,label):
        # Data is required to be given in a three dimensional numpy array, [nDocuments,nVocabulary,nObserved]
        # Label is required to be given in a numpy array, [nDocuments,]
        
        # Set additional parameters
        self.nDocuments = data.shape[0]
        self.nVocabulary = data.shape[1]
        self.data = data
        self.label = label

    def save(self,name):
        # Save ObLect
        with open(name,"wb") as output:
            cPickle.dump(self.__dict__,output,protocol=cPickle.HIGHEST_PROTOCOL)

    def load(self,name):
        # Load Object
        with open(name,"rb") as input:
            self.__dict__.update(cPickle.load(input))

    def sumMulti(self,doc,qZ,eta,nd):
        return np.prod(np.power(np.dot(qZ,np.exp(eta/nd)),doc[:,1]))

    def sumMultiOmit(self,doc,qZ,eta,nd):
        return np.full(qZ.shape[0],self.sumMulti(doc,qZ,eta,nd))/(np.dot(qZ,np.exp(eta/nd)))
        
    def L(self,eta,*args):
        # Return cost function to minimize
        lEta = 0.0
        data,qZAll,zetaAll,labelAll = args
        for d in range(data.shape[0]):
            doc = data[d,:,:]
            qZ = qZAll[d,:,:]
            zeta = zetaAll[d]
            label = labelAll[d]
            nd = float(doc[:,1].sum())
            lEta += label/nd*np.dot(np.dot(qZ,eta),doc[:,1]) - 1.0/zeta*self.sumMulti(doc,qZ,eta,nd)
        return -lEta

    def gradient(self,eta,*args):
        # Return gradient for cost function
        data,qZAll,zetaAll,labelAll = args
        lEtaGradient = np.zeros(eta.shape[0])
        for d in range(data.shape[0]):
            doc = data[d,:,:]
            qZ = qZAll[d,:,:]
            zeta = zetaAll[d]
            label = labelAll[d]
            nd = float(doc[:,1].sum())
            buff = label*(qZ*doc[:,1].reshape(qZ.shape[0],1)).sum(axis=0)
            buff += -1.0/zeta*((qZ*np.exp(eta/nd).reshape(1,qZ.shape[1])*doc[:,1].reshape(qZ.shape[0],1))*self.sumMultiOmit(doc,qZ,eta,nd).reshape(qZ.shape[0],1)).sum(axis=0)
            lEtaGradient += buff/nd
        return -lEtaGradient

    def solve(self,nTopics=8,epsilon=1e-5,alpha=1.0,beta=0.01):

        # Set additional parameters
        self.nTopics = nTopics
        self.epsilon = epsilon
        
        # Prior distribution for alpha and beta
        self.alpha = np.full(self.nTopics,alpha,dtype=np.float64)
        self.beta = np.full(self.nVocabulary,beta,dtype=np.float64)
        
        # Define q(theta)
        self.qTheta = np.empty((self.nDocuments,self.nTopics),dtype=np.float64)
        self.qThetaNew = np.empty((self.nDocuments,self.nTopics),dtype=np.float64)
        
        # Define q(phi)
        self.qPhi = np.empty((self.nTopics,self.nVocabulary),dtype=np.float64)
        
        # Initialize q(z)
        self.qZ = np.random.rand(self.nDocuments,self.nVocabulary,self.nTopics)
        for i in range(self.qZ.shape[0]):
            self.qZ[i] /= self.qZ[i].sum(axis=1).reshape((self.qZ[i].shape[0],1))

        # Define eta
        self.eta = np.full(self.nTopics,1.0,dtype=np.float64)

        # Define zeta
        self.zeta = np.full(self.nDocuments,1.0,dtype=np.float64)
        
        # Start solving using variational Bayes
        nIteration = 0
        while(1):
            
            deltaMax = 0.0
            tic = time.clock()

            # Update qPhi
            qPhi = self.qPhi[:,:]
            qPhi[:] = np.tile(self.beta.reshape((1,self.nVocabulary)),(self.nTopics,1))-1.0
            for d in range(self.nDocuments):
                doc = self.data[d,:,:]
                qZ = self.qZ[d,:,:]
                qPhi += (qZ[:,:] * doc[:,1].reshape((doc.shape[0],1))).T
            phiExpLog = scipy.special.psi(self.qPhi[:,:]+1.0)
            phiExpLog -= np.tile(scipy.special.psi((self.qPhi[:,:]+1.0).sum(axis=1)).reshape((self.nTopics,1)),(1,self.nVocabulary))

            # Update eta
            eta = self.eta[:]
            eta[:] = optimize.fmin_cg(self.L,eta,fprime=self.gradient,args=(self.data,self.qZ,self.zeta,self.label))
            
            # Iterate documents
            for d in range(self.nDocuments):
                
                doc = self.data[d,:,:]
                qZ = self.qZ[d,:,:]
                qTheta = self.qTheta[d,:]
                qThetaNew = self.qThetaNew[d,:]
                label = self.label[d]
                nd = float(doc[:,1].sum())
                
                # Update qTheta
                if nIteration == 0:
                    qTheta[:] = self.alpha-1.0
                    qTheta += (qZ * doc[:,1].reshape((qZ.shape[0],1))).sum(axis=0)
                else:
                    qTheta[:] = qThetaNew
                thetaExpLog = scipy.special.psi(qTheta+1.0)
                thetaExpLog -= scipy.special.psi((qTheta+1.0).sum())

                # Update zeta
                self.zeta[d] = self.sumMulti(doc,qZ,eta,nd) + np.dot(qZ.sum(axis=1),doc[:,1])
                
                # Update qZ
                lq = np.tile(self.eta*label/nd,(self.nVocabulary,1))
                lq -= np.tile(1.0/self.zeta[d]*np.exp(eta/nd),(self.nVocabulary,1))*self.sumMultiOmit(doc,qZ,eta,nd).reshape((self.nVocabulary,1))
                qZ[:,:] = np.exp(phiExpLog.T+np.tile(thetaExpLog.reshape((1,self.nTopics)),(self.nVocabulary,1))+lq)
                qZ /= qZ.sum(axis=1).reshape((self.nVocabulary,1))
                
                # Measure amount of change
                qThetaNew[:] = self.alpha-1.0
                qThetaNew += (qZ * doc[:,1].reshape((qZ.shape[0],1))).sum(axis=0)
                delta = np.abs(qTheta-qThetaNew).sum()/doc[:,1].sum()
                deltaMax = max(deltaMax,delta)
            
            # Break if converged
            if deltaMax<self.epsilon:
                break

            toc = time.clock()
            print (nIteration,deltaMax,toc-tic)
            print eta
            nIteration += 1         
        return

    def predict(self,dataPredict):
        # Data to predict is required to be given in a three dimensional numpy array, [nDocuments,nVocabulary,nObserved]

         # Utilize topic information with training data
        phiExpLog = scipy.special.psi(self.qPhi[:,:]+1.0)
        phiExpLog -= np.tile(scipy.special.psi((self.qPhi[:,:]+1.0).sum(axis=1)).reshape((self.nTopics,1)),(1,self.nVocabulary))
        
        # Define and Initialize parameters
        nDataPredict = dataPredict.shape[0]
        qThetaPredict = np.empty((nDataPredict,self.nTopics),dtype=np.float64)
        qThetaPredictNew = np.empty((nDataPredict,self.nTopics),dtype=np.float64)
        qZPredict = np.random.rand(nDataPredict,self.nVocabulary,self.nTopics)
        qZetaPredict = np.full(nDataPredict,10,dtype=np.float64)
        for i in range(qZPredict.shape[0]):
            qZPredict[i] /= qZPredict[i].sum(axis=1).reshape((qZPredict[i].shape[0],1))
            
        # Start predicting
        nIteration = 0
        while(1):

            deltaMax = 0.0
            tic = time.clock()

            # Iterate documents
            for d in range(nDataPredict):
                
                doc = dataPredict[d,:,:]
                qZ = qZPredict[d,:,:]
                qTheta = qThetaPredict[d,:]
                qThetaNew = qThetaPredictNew[d,:]
                nd = float(doc[:,1].sum())
                
                # Update qTheta
                if nIteration == 0:
                    qTheta[:] = self.alpha-1.0
                    qTheta += (qZ * doc[:,1].reshape((qZ.shape[0],1))).sum(axis=0)
                else:
                    qTheta[:] = qThetaNew
                thetaExpLog = scipy.special.psi(qTheta+1.0)
                thetaExpLog -= scipy.special.psi((qTheta+1.0).sum())

                # Update zeta
                qZetaPredict[d] = self.sumMulti(doc,qZ,self.eta,nd) + np.dot(qZ.sum(axis=1),doc[:,1])
                
                # Update qZ
                qZ[:,:] = np.exp(phiExpLog.T+np.tile(thetaExpLog.reshape((1,self.nTopics)),(self.nVocabulary,1)))
                qZ /= qZ.sum(axis=1).reshape((self.nVocabulary,1))
                
                # Measure amount of change
                qThetaNew[:] = self.alpha-1.0
                qThetaNew += (qZ * doc[:,1].reshape((qZ.shape[0],1))).sum(axis=0)
                delta = np.abs(qTheta-qThetaNew).sum()/doc[:,1].sum()
                deltaMax = max(deltaMax,delta)
            
            # Break if converged
            if deltaMax<self.epsilon:
                break

            toc = time.clock()
            print (nIteration,deltaMax,toc-tic)
            nIteration += 1

        # Logistic Regression
        predictedLabel = np.zeros(0)
        for d in range(nDataPredict):
            buff = np.dot(self.eta,(qZPredict[d,:,:]*dataPredict[d,:,1].reshape((self.nVocabulary,1))).sum(axis=0)/dataPredict[d,:,1].sum())
            predictedLabel = np.append(predictedLabel,np.exp(buff-np.log(1.0+np.exp(buff))))

        return predictedLabel
    








