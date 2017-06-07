# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1])/100

    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
<<<<<<< HEAD
        
        for x in range(self.trainingSet.input.shape[0]):
            if self.trainingSet.label[x] != 1:
                self.trainingSet.input[x,:] *= -1
        
        for y in range(self.epochs):
            outputvecbool = self.classify(self.trainingSet.input)
            errorsumvec = 0
        
            for x in range(self.trainingSet.input.shape[0]):
                if outputvecbool[x] != 1:
                    errorsumvec += self.trainingSet.input[x,:]
    
            self.updateWeights(errorsumvec)
                #ashhs
=======

        # Try to use the abstract way of the framework
        from util.loss_functions import DifferentError
        loss = DifferentError()

        learned = False
        iteration = 0

        # Train for some epochs if the error is not 0
        while not learned:
            totalError = 0
            for input, label in zip(self.trainingSet.input,
                                    self.trainingSet.label):
                output = self.fire(input)
                if output != label:
                    error = loss.calculateError(label, output)
                    self.updateWeights(input, error)
                    totalError += error

            iteration += 1
            
            if verbose:
                logging.info("Epoch: %i; Error: %i", iteration, -totalError)
            
            if totalError == 0 or iteration >= self.epochs:
                # stop criteria is reached
                learned = True
>>>>>>> 30206912729d03608f6a3e8e44b92b40f2b83a5d

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
<<<<<<< HEAD
        
=======
>>>>>>> 30206912729d03608f6a3e8e44b92b40f2b83a5d
        return self.fire(testInstance)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

<<<<<<< HEAD
    def updateWeights(self, error):
    
        # Write your code to update the weights of the perceptron here
        self.weight += error*self.learningRate
         
=======
    def updateWeights(self, input, error):
        self.weight += self.learningRate*error*input

>>>>>>> 30206912729d03608f6a3e8e44b92b40f2b83a5d
    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        return Activation.sign(np.dot(np.array(input), self.weight))
