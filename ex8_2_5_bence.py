# exercise 8.2.5
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
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

#get row 9 (chd) name for the output
outputVariableName = attributeNames.pop(9)

#convert dataframe to nparray and remove column for row.names
data = np.array(heart_data.values, dtype=np.float64)
data = np.delete (data,[0],1)

#get all columns for x except column 9 (chd)
X = np.delete(data,[9],1)

#load column 9 (chd) into y
y = np.delete(data,[0,1,2,3,4,5,6,7,8],1)

#Downsample: X = X[1:20,:] y = y[1:20,:]
N, M = X.shape
C = 2

# Normalize data
X = stats.zscore(X);

AllEstimates = []
AllControls = []
AllWeights = []
AllErrors = []
AllBiases = []
constantTF = ['Tanh()', 'Sigmoid()']
constantAttNames = ['sbp',
 'tobacco',
 'ldl',
 'adiposity',
 'famhist',
 'typea',
 'obesity',
 'alcohol',
 'age']

def doEverything(k):
    # Parameters for neural network classifier
    n_hidden_units = k     # number of hidden units
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 30000         # stop criterion 2 (max epochs in training)
    
    # K-fold crossvalidation
    K = 5                  # only five folds to speed up this example
    CV = model_selection.KFold(K, shuffle=True, random_state=240)
    # Make figure for holding summaries (errors and learning curves)
    summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
    # Make a list for storing assigned color of learning curve for up to K=10
    color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
                  'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']
    
    # Define the model, see also Exercise 8.2.2-script for more information.
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(n_hidden_units, 1), # H hidden units to 1 output neuron
                        torch.nn.Sigmoid() # final tranfer function
                        )
    loss_fn = torch.nn.BCELoss()

    print('Training model of type:\n\n{}\n'.format(str(model())))
    errors = [] # make a list for storing generalizaition error in each loop
    for k, (train_index, test_index) in enumerate(CV.split(X,y)): 
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
        
        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index,:])
        y_test = torch.Tensor(y[test_index])
        
        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(model,
                                                           loss_fn,
                                                           X=X_train,
                                                           y=y_train,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter,
                                                           tolerance=1e-8,
                                                           logging_frequency=30000)
        
        print('\n\tBest loss: {}\n'.format(final_loss))
        
        # Determine estimated class labels for test set
        y_sigmoid = net(X_test)
        y_test_est = (y_sigmoid>.5).type(dtype=torch.uint8)
    
        # Determine errors and errors
        y_test = y_test.type(dtype=torch.uint8)
    
        e = y_test_est != y_test
        error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
        errors.append(error_rate) # store error rate for current CV fold 
        
        #save age estimates for later plotting
        AllEstimates.append(y_test_est.data.numpy())
        AllControls.append(y_test.data.numpy())
        AllWeights.append([net[i].weight.data.numpy().T for i in [0,2]])
        AllBiases.append([net[i].bias.data.numpy() for i in [0,2]])
        
        # Display the learning curve for the best net in the current fold
        h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
        h.set_label('CV fold {0}'.format(k+1))
        summaries_axes[0].set_xlabel('Iterations')
        summaries_axes[0].set_xlim((0, max_iter))
        summaries_axes[0].set_ylabel('Loss')
        summaries_axes[0].set_title('Learning curves')
        
    
    AllErrors.append(errors)
    # Display the error rate across folds
    summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
    summaries_axes[1].set_xlabel('Fold');
    summaries_axes[1].set_xticks(np.arange(1, K+1))
    summaries_axes[1].set_ylabel('Error rate');
    summaries_axes[1].set_title('Test misclassification rates')
    
# =============================================================================
#     print('Diagram of best neural net in last fold:')
#     weights = [net[i].weight.data.numpy().T for i in [0,2]]
#     biases = [net[i].bias.data.numpy() for i in [0,2]]
#     tf =  [str(net[i]) for i in [1,3]]
#     draw_neural_net(weights, biases, tf, attribute_names=attributeNames)
# =============================================================================
    
    # Print the average classification error rate
    print('\nGeneralization error/average error rate: {0}%'.format(round(100*np.mean(errors),4)))

def visualize(ageEstimates,ageControls, biases, errors, weights):
        
    print('Diagram of best neural net in last fold:')
    tf =  constantTF
    draw_neural_net(weights, biases, tf, attribute_names=constantAttNames)
    
    # Print the average classification error rate
    print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))
    
    # When dealing with regression outputs, a simple way of looking at the quality
    # of predictions visually is by plotting the estimated value as a function of 
    # the true/known value - these values should all be along a straight line "y=x", 
    # and if the points are above the line, the model overestimates, whereas if the
    # points are below the y=x line, then the model underestimates the value
    plt.figure(figsize=(10,10))
    y_est = ageEstimates; y_true = ageControls
    
    # =============================================================================
    # axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
    #
    # axis range changed to dataset's range so the plots share proportions
    # =============================================================================
    axis_range = [np.min([y_true])-1,np.max([y_true])+1]
    
    plt.plot(axis_range,axis_range,'k--')
    plt.plot(y_true, y_est,'ob',alpha=.25)
    plt.legend(['Perfect estimation','Model estimations'])
    plt.title('Alcohol content: estimated versus true value (for last CV-fold)')
    plt.ylim(axis_range); plt.xlim(axis_range)
    plt.xlabel('True value')
    plt.ylabel('Estimated value')
    plt.grid()
    
    plt.show()
    
    print('Ran Exercise 8.2.5')
    
def vizHandler(node, foldNo):
    folds=5
    visualize(AllEstimates[node*folds+foldNo],
              AllControls[node*folds+foldNo],
              AllBiases[node*folds+foldNo],
              AllErrors[node],
              AllWeights[node*folds+foldNo])

print('Ran Exercise 8.2.5')