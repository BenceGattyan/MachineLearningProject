# exercise 8.1.1

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rlr_validate
import pandas as pd


filename = r'C:\Users\User\OneDrive\School\MachineLearning&DataMining\Project1\ahd.csv'
heart_data = pd.read_csv(filename)
## Cleaning up data
# We notice that there is a binary attribute "famhist" that describes whether 
# a patient has a family history of heart disease or not. We will convert the
# string values "Absent" and "Present" to "0" and "1"
heart_data.famhist = heart_data.famhist.str.replace('Absent', '0')
heart_data.famhist = heart_data.famhist.str.replace('Present', '1')


attributeNames = ["sbp","tobacco","ldl","adiposity","famhist","typea","obesity","alcohol","chd"]
#print(attributes)


## X,y-format
# If the modelling problem of interest was a classification problem where
# we wanted to classify the chd response attribute, we could now identify obtain
# the data in the X,y-format as so:
data = np.array(heart_data.values, dtype=np.float64)

# Extracing the X matrix from the data set (every column exlcuding chd response)
X = np.delete(data, [1,9], 1)

# chd response attrbiute will be the y-vector
y = data[:,9]


#N = X.shape[0]
#C = X.shape[1]

#mat_data = loadmat('../Data/body.mat')
#X = mat_data['X']
#y = mat_data['y'].squeeze()
#attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

#remove freatures
#X=np.multiply(X,0)

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV = model_selection.KFold(K, shuffle=True, random_state=240)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
#lambdas = np.power(10.,np.arange(-1,7,0.25))

# Values of lambda
lambda_interval = np.logspace(-1, 6, 50)

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
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
for (o,(train_index, test_index)) in enumerate(CV.split(X,y)):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index] 
    internal_cross_validation=10
    
    mdl = LogisticRegression(penalty='l2',
                             C=1/lambda_interval[k],
                             max_iter=30000,
                             tol=1e-8,
                             random_state=240)
    
    mdl.fit(X_train, y_train)
 
    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T
    
    train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)
 
    w_est = mdl.coef_[0] 
    coefficient_norm[k] = np.sqrt(np.sum(w_est**2))

    if k==K-1:
        min_error = np.min(test_error_rate)
        opt_lambda_idx = np.argmin(test_error_rate)
        opt_lambda = lambda_interval[opt_lambda_idx]
        
        subplot(1,1,1)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambda_interval,train_error_rate.T,'b.-',lambda_interval,test_error_rate.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    k+=1

show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train_nofeatures.mean()))
print('- Test error:     {0}'.format(Error_test_nofeatures.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))

print('Ran Exercise 8.1.1')
