# -*- coding: utf-8 -*-

import numpy as np
import scipy.special
import time, cPickle
from scipy import optimize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class sLDA:
    # variational implimentation of supervised LDA

    def __init__(self):
        # do nothing particularly
        pass

    def setData(self,data,label):
        # data is required to be given in a two dimensional numpy array (nDocuments,nVocabulary)
        # with each element representing the number of times observed
        # label is required to be given in an one dimensional numpy array (nDocuments,)
        
        # set parameters
        self.data = data
        self.label = label
        self.nDocuments = data.shape[0]
        self.nVocabulary = data.shape[1]

    def sumMulti(self,doc,qZ,eta,nd):
        return np.prod(np.power(np.dot(qZ,np.exp(eta/nd)),doc))

    def sumMultiOmit(self,doc,qZ,eta,nd):
        return np.full(qZ.shape[0],self.sumMulti(doc,qZ,eta,nd))/(np.dot(qZ,np.exp(eta/nd)))
        
    def costFunction(self,eta,*args):
        # return cost function to minimize
        lEta = 0.0
        data,qZAll,zetaAll,labelAll = args

        # iterate over all documents
        for d in range(data.shape[0]):
            doc = data[d,:]
            qZ = qZAll[d,:,:]
            zeta = zetaAll[d]
            label = labelAll[d]
            nd = float(doc.sum())
            lEta += label/nd*np.dot(np.dot(qZ,eta),doc) - 1.0/zeta*self.sumMulti(doc,qZ,eta,nd)
        return -lEta

    def gradient(self,eta,*args):
        # return gradient for cost function
        data,qZAll,zetaAll,labelAll = args
        lEtaGradient = np.zeros(eta.shape[0])

        # iterate over all documents
        for d in range(data.shape[0]):
            doc = data[d,:]
            qZ = qZAll[d,:,:]
            zeta = zetaAll[d]
            label = labelAll[d]
            nd = float(doc.sum())
            buff = label*(qZ*doc.reshape(qZ.shape[0],1)).sum(axis=0)
            buff += -1.0/zeta*((qZ*np.exp(eta/nd).reshape(1,qZ.shape[1])*doc.reshape(qZ.shape[0],1))*self.sumMultiOmit(doc,qZ,eta,nd).reshape(qZ.shape[0],1)).sum(axis=0)
            lEtaGradient += buff/nd
        return -lEtaGradient

    def solve(self,nTopics=8,epsilon=1e-5,alpha=1.0,beta=0.01):

        # set additional parameters
        self.nTopics = nTopics
        self.epsilon = epsilon
        
        # prior distribution for alpha and beta
        self.alpha = np.full(self.nTopics,alpha,dtype=np.float64)
        self.beta = np.full(self.nVocabulary,beta,dtype=np.float64)
        
        # define q(theta)
        self.qTheta = np.empty((self.nDocuments,self.nTopics),dtype=np.float64)
        self.qThetaNew = np.empty((self.nDocuments,self.nTopics),dtype=np.float64)
        
        # define q(phi)
        self.qPhi = np.empty((self.nTopics,self.nVocabulary),dtype=np.float64)
        
        # dfine and initialize q(z)
        self.qZ = np.random.rand(self.nDocuments,self.nVocabulary,self.nTopics)
        for i in range(self.qZ.shape[0]):
            self.qZ[i] /= self.qZ[i].sum(axis=1).reshape((self.qZ[i].shape[0],1))

        # Define and initialize eta
        self.eta = np.full(self.nTopics,1.0,dtype=np.float64)

        # Define and initialize zeta
        self.zeta = np.full(self.nDocuments,1.0,dtype=np.float64)
        
        # start solving using variational Bayes
        nIteration = 0
        while(1):
            
            deltaMax = 0.0
            tic = time.clock()

            # Update qPhi
            qPhi = self.qPhi[:,:]
            qPhi[:] = np.tile(self.beta.reshape((1,self.nVocabulary)),(self.nTopics,1))
            for d in range(self.nDocuments):
                doc = self.data[d,:]
                qZ = self.qZ[d,:,:]
                qPhi += (qZ[:,:] * doc.reshape((doc.shape[0],1))).T
            phiExpLog = scipy.special.psi(self.qPhi[:,:])
            phiExpLog -= np.tile(scipy.special.psi((self.qPhi[:,:]).sum(axis=1)).reshape((self.nTopics,1)),(1,self.nVocabulary))

            # Update eta
            eta = self.eta[:]
            eta[:] = optimize.fmin_cg(self.costFunction,eta,fprime=self.gradient,args=(self.data,self.qZ,self.zeta,self.label))
            
            # Iterate documents
            for d in range(self.nDocuments):
                
                doc = self.data[d,:]
                qZ = self.qZ[d,:,:]
                qTheta = self.qTheta[d,:]
                qThetaNew = self.qThetaNew[d,:]
                label = self.label[d]
                nd = float(doc.sum())
                
                # update qTheta
                if nIteration == 0:
                    qTheta[:] = self.alpha
                    qTheta += (qZ * doc.reshape((qZ.shape[0],1))).sum(axis=0)
                else:
                    qTheta[:] = qThetaNew
                thetaExpLog = scipy.special.psi(qTheta)
                thetaExpLog -= scipy.special.psi((qTheta).sum())

                # update zeta
                self.zeta[d] = self.sumMulti(doc,qZ,eta,nd) + np.dot(qZ.sum(axis=1),doc)
                
                # update qZ
                lq = np.tile(self.eta*label/nd,(self.nVocabulary,1))
                lq -= np.tile(1.0/self.zeta[d]*np.exp(eta/nd),(self.nVocabulary,1))*self.sumMultiOmit(doc,qZ,eta,nd).reshape((self.nVocabulary,1))
                qZ[:,:] = np.exp(phiExpLog.T+np.tile(thetaExpLog.reshape((1,self.nTopics)),(self.nVocabulary,1))+lq)
                qZ /= qZ.sum(axis=1).reshape((self.nVocabulary,1))
                
                # measure amount of change
                qThetaNew[:] = self.alpha
                qThetaNew += (qZ * doc.reshape((qZ.shape[0],1))).sum(axis=0)
                delta = np.abs(qTheta-qThetaNew).sum()/doc.sum()
                deltaMax = max(deltaMax,delta)
            
            # break if converged
            if deltaMax<self.epsilon:
                break

            # display information
            toc = time.clock()
            self.heatmap(nIteration)
            print "nIteration=%d, delta=%f, time=%.5f"%(nIteration,deltaMax,toc-tic)
            print eta
            nIteration += 1

    def predict(self,dataPredict):
        # dataPredict is required to be given in a two dimensional numpy array (nDocuments,nVocabulary)
        # with each element representing the number of times observed

        # set additional parameters
        nDataPredict = dataPredict.shape[0]

         # utilize topic information with training data
        phiExpLog = scipy.special.psi(self.qPhi[:,:])
        phiExpLog -= np.tile(scipy.special.psi((self.qPhi[:,:]).sum(axis=1)).reshape((self.nTopics,1)),(1,self.nVocabulary))
        
        # define q(theta) for unseen data
        qThetaPredict = np.empty((nDataPredict,self.nTopics),dtype=np.float64)
        qThetaPredictNew = np.empty((nDataPredict,self.nTopics),dtype=np.float64)

        # define and initialize q(z) for unseend data
        qZPredict = np.random.rand(nDataPredict,self.nVocabulary,self.nTopics)
        for i in range(qZPredict.shape[0]):
            qZPredict[i] /= qZPredict[i].sum(axis=1).reshape((qZPredict[i].shape[0],1))
        
        # define and initialize zeta for unseen data
        qZetaPredict = np.full(nDataPredict,10,dtype=np.float64)

        # start prediction
        nIteration = 0
        while(1):

            deltaMax = 0.0
            tic = time.clock()

            # iterate over all documents
            for d in range(nDataPredict):
                doc = dataPredict[d,:]
                qZ = qZPredict[d,:,:]
                qTheta = qThetaPredict[d,:]
                qThetaNew = qThetaPredictNew[d,:]
                nd = float(doc.sum())
                
                # update qTheta for unseen data
                if nIteration == 0:
                    qTheta[:] = self.alpha
                    qTheta += (qZ * doc.reshape((qZ.shape[0],1))).sum(axis=0)
                else:
                    qTheta[:] = qThetaNew
                thetaExpLog = scipy.special.psi(qTheta)
                thetaExpLog -= scipy.special.psi((qTheta).sum())

                # update zeta for unseen data
                qZetaPredict[d] = self.sumMulti(doc,qZ,self.eta,nd) + np.dot(qZ.sum(axis=1),doc)
                
                # update qZ for unseen data
                qZ[:,:] = np.exp(phiExpLog.T+np.tile(thetaExpLog.reshape((1,self.nTopics)),(self.nVocabulary,1)))
                qZ /= qZ.sum(axis=1).reshape((self.nVocabulary,1))
                
                # measure amount of change
                qThetaNew[:] = self.alpha
                qThetaNew += (qZ * doc.reshape((qZ.shape[0],1))).sum(axis=0)
                delta = np.abs(qTheta-qThetaNew).sum()/doc.sum()
                deltaMax = max(deltaMax,delta)
            
            # break if converged
            if deltaMax<self.epsilon:
                break

            # display information
            toc = time.clock()
            print (nIteration,deltaMax,toc-tic)
            nIteration += 1

        # apply logistic regression
        predictedLabel = np.zeros(0)
        for d in range(nDataPredict):
            buff = np.dot(self.eta,(qZPredict[d,:,:]*dataPredict[d,:].reshape((self.nVocabulary,1))).sum(axis=0)/dataPredict[d,:].sum())
            predictedLabel = np.append(predictedLabel,np.exp(buff-np.log(1.0+np.exp(buff))))

        return predictedLabel

    def heatmap(self,nIteration):
        # save heatmap image of topic-word distribution
        topicWordDistribution = self.qPhi/self.qPhi.sum(axis=1).reshape((self.nTopics,1))

        plt.clf()
        fig,ax = plt.subplots()

        # visualize topic-word distribution
        X,Y = np.meshgrid(np.arange(topicWordDistribution.shape[1]+1),np.arange(topicWordDistribution.shape[0]+1))
        image = ax.pcolormesh(X,Y,topicWordDistribution)
        plt.xlim(0,topicWordDistribution.shape[1])
        plt.xlabel("Vocabulary ID")
        plt.ylabel("Topic ID")
        plt.yticks(np.arange(self.nTopics+1)+0.5,self.eta)

        # show colorbar
        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="2%",pad=0.05)
        fig.add_axes(ax_cb)
        plt.colorbar(image,cax=ax_cb)
        figure = plt.gcf()
        figure.set_size_inches(16,12)
        plt.tight_layout()

        # save image as a file
        plt.savefig("visualization/nIteration_%d.jpg"%nIteration,dpi=100)
        plt.close()

    def save(self,name):
        # save object as a file
        with open(name,"wb") as output:
            cPickle.dump(self.__dict__,output,protocol=cPickle.HIGHEST_PROTOCOL)

    def load(self,name):
        # load object from a file
        with open(name,"rb") as input:
            self.__dict__.update(cPickle.load(input))







