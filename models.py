import numpy as np
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC


class UniformOVA(BaseEstimator):
    '''
    UniformOVA estimator fits a linear SVC model for each class that has
    data, and a NullModel for any classes with no positive instances.
    It predicts class membership whenever the decision value is either above
    one threshold or within a second threshold of the highest value 
    for that instance.
    '''

    # NB: best known values are c=1 (default), t1=-0.3, t2=0.1
    def __init__(self, c=1, t1=0, t2=0, null_dv=-99):
        '''
        Constructor for UniformOVA model. Just stores field values.
        
        Params:
          t1 - either a scalar threshold, or a vector of length(dv.shape[1])
             all instances with dvs > t1 are positive
          t2 - all instances with dvs >= row_max - t2 are positive
          c - L2 loss parameter for the SVC's
          null_dv - the decision value for classes with no positive instances.
          
        Returns:
          an initialized UniformOVA model
        '''
        self.t1 = t1
        self.t2 = t2
        self.c = c
        self.null_dv = null_dv

    def fit(self, x, y):
        '''
        Fit the UniformOVA model.
        
        Params:
          x - input features
          y - 0-1 label matrix
      
        Returns:
          nothing, but model is fitted.
        '''
        self.models = []
        for k in range(y.shape[1]):
            if (y[:, k]).any():
                model = LinearSVC(C=self.c)
                model.fit(x, y[:, k])
            else:
                model = NullModel(self.null_dv)
            self.models.append(model)

    def predict(self, x):
        '''
        Prediction method predicts class membership of instances with decision 
        values above threshold t1 or within t2 of the highest decision value 
        on that instance.
        
        Params:
          x - input features, not used
          y - 0-1 label matrix, not used
      
        Returns:
          A 0-1 matrix of predicted labels of size (# instances) x (# classes).
        '''
        dvs = self.decision_function(x)
        pred = (dvs > self.t1).astype(float)
        max_dv = dvs.max(1)
        for k in range(pred.shape[0]):
            cut = max_dv[k] - self.t2
            idx = (dvs[k, :] >= cut)
            pred[k, idx] = 1
        return pred

    def decision_function(self, x):
        '''
        Finds the decision value for each instance under each per-class model.
        
        Params:
          x - input features, not used
        
        Returns:
          a real-valued matrix of dimension (# instances) x (# classes)
        '''
        dvs = np.zeros((x.shape[0], len(self.models)))
        for k in range(len(self.models)):
            dvs[:, k] = self.models[k].decision_function(x)
        return dvs


class NullModel(BaseEstimator):
    '''
    NullModel returns a decision value that results in a negative prediction.
    It is used for the 3 classes that do not appear in the training data.
    This model allows us to just keep a list of models for all of the classes.
    Normal models can't be fitted on classes with only one label. Unlike the 
    other models, NullModel is for only one class.
    '''

    def __init__(self, null_dv=-99):
        '''
        Constructor stores the constant decision value to use.
        
        Params:
          null_dv - the decision value to return
          
        Returns: a NullModel
        '''
        self.null_dv = null_dv

    def fit(self, x, y):
        '''
        Fit is a no-op for the NullModel
        
        Params:
          x - input features, not used
          y - 0-1 label vector
        
        Returns: nothing
        '''
        pass

    def predict(self, x):
        '''
        For NullModel, predict() always returns 0 (non-membership).
        
        Params:
          x - input features, not used
          
        Returns:
          0, always
        '''
        return 0

    def decision_function(self, x):
        '''
        Returns the null_dv for all instances.
        
        Params:
          x - input features, not used
          
        Returns:
         the null_dv, always
        '''
        return self.null_dv * np.ones(x.shape[0])
