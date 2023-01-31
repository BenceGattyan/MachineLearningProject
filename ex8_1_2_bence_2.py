# exercise 8.1.1

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rlr_validate
import pandas as pd


# Load Heart Disease data
filename = r'C:/Users/User/OneDrive/School/MachineLearning&DataMining/Project1/ahd.csv'
heart_data = pd.read_csv(filename)

# string values "Absent" and "Present" to "0" and "1"
heart_data.famhist = heart_data.famhist.str.replace('Absent', '0')
heart_data.famhist = heart_data.famhist.str.replace('Present', '1')

#Get attribute names and remove row.names
attributeNames = list(heart_data.columns.values)
attributeNames.pop(0)

#get row 9 (chd response) name for the output
outputVariableName = attributeNames.pop(9)

#convert dataframe to nparray and remove column for row.names
data = np.array(heart_data.values, dtype=np.float64)
data = np.delete (data,[0],1)

#get all columns for x except column 9 (chd response)
X = np.delete(data,[9],1)

#load column 9 (chd response) into y
#y = np.delete(data,[0,1,2,3,4,5,6,7,8],1)
y = data[:,9]

classNames=['Negative','Positive']

# =============================================================================
# mat_data = loadmat('../Data/body.mat')
# X = mat_data['X']
# y = mat_data['y'].squeeze()
# attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
# N, M = X.shape
# 
# # Add offset attribute
# X = np.concatenate((np.ones((X.shape[0],1)),X),1)
# attributeNames = [u'Offset']+attributeNames
# =============================================================================

N, M = X.shape
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV = model_selection.KFold(K, shuffle=True, random_state=240)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambda_interval = np.logspace(0, 0.2, 3)

# Initialize variables
#T = len(lambdas)
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))

minTestErrorsValues = []
minTestErrors = []
minTestErrorsL2 = []

k=0
for train_index, test_index in CV.split(X,y):
    for k in range(0, len(lambda_interval)):
        # extract training and test set for current CV fold
        X_train = X[train_index]*1
        y_train = y[train_index]
        X_test = X[test_index]*1
        y_test = y[test_index]
        internal_cross_validation = 10    
        
        mdl = LogisticRegression(penalty='l2',
                                 C=1/lambda_interval[k],
                                 max_iter=1000,
                                 tol=1e-8,
                                 random_state=240)
        
        mdl.fit(X_train, y_train)
    
        y_train_est = mdl.predict(X_train).T
        y_test_est = mdl.predict(X_test).T
        
        train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
        test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)
    
        w_est = mdl.coef_[0] 
        weights = mdl.coef_
        coefficient_norm[k] = np.sqrt(np.sum(w_est**2))
    
        k+=1    
    
    # Display the results for the last cross-validation fold
    if True:
        min_error = np.min(test_error_rate)
        opt_lambda_idx = np.argmin(test_error_rate)
        opt_lambda = lambda_interval[opt_lambda_idx]
        
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        plt.semilogx(lambda_interval, train_error_rate*100)
        plt.semilogx(lambda_interval, test_error_rate*100)
        plt.semilogx(opt_lambda, min_error*100, 'o')
        plt.text(min(lambda_interval),15, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
        plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
        plt.ylabel('Error rate (%)')
        plt.title('Classification error')
        plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
        plt.ylim([0, 50])
        plt.grid()
        
        minTestErrorsValues.append(np.round(min_error*100,2))
        minTestErrors.append(np.round(opt_lambda,2))
        #minTestErrorsL2
        
        #plt.figure(figsize=(8,8))
        subplot(1,2,2)
        plt.semilogx(lambda_interval, coefficient_norm,'k')
        plt.ylabel('L2 Norm')
        plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
        plt.title('Parameter vector L2 norm')
        plt.grid()
        grid()
    
        show()
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))
    
        





# =============================================================================
# # Display results
# print('Linear regression without feature selection:')
# print('- Training error: {0}'.format(Error_train.mean()))
# print('- Test error:     {0}'.format(Error_test.mean()))
# print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
# print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
# print('Regularized linear regression:')
# print('- Training error: {0}'.format(Error_train_rlr.mean()))
# print('- Test error:     {0}'.format(Error_test_rlr.mean()))
# print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
# print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))
# 
# print('Weights in last fold:')
# for m in range(M):
#     print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))
# =============================================================================

print('Ran Exercise 8.1.1')