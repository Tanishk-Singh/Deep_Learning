
# ## Perceptron as a linear classifier Unit

# In[1]:


import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs, make_moons


# In[2]:


X,Y = make_blobs(n_samples=500,
    n_features=2,
    centers=2,
    cluster_std=1.0,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=1,)

# X,Y = make_moons(n_samples=1000, shuffle=True, noise=0.2 , random_state=None)


# In[3]:


print(X.shape, Y.shape)


# In[4]:


print(Y[:5])


# In[5]:


print(X[:5])


# In[6]:


plt.style.use('seaborn')
plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.Accent)
plt.show()


# ## Model and Helper Functions

# In[7]:


def sigmoid(z):
    return (1.0)/(1+np.exp(-z))


# In[8]:


sigmoid(5)


### Implement Perceptron Learning Algorithm

# In[9]:


def predict(X, weights):
    """ X -> (m x (n+1)) matrix, w - ((n+1)x1)>"""
    z = np.dot(X, weights)
    predictions  = sigmoid(z)
    return predictions


def loss (X,Y,weights):
    """Binary Cross Entropy"""
    Y_ = predict(X,weights)
    cost = np.mean(-Y*np.log(Y_)-(1-Y)*np.log(1-Y_))
    return cost

def update(X,Y,weights,learning_rate):
    """Perform weight updates for 1 epoch"""
    
    Y_ = predict(X,weights)
    dw = np.dot(X.T,Y_-Y)
    
    m = X.shape[0]
    
    weights = weights - learning_rate*dw/(float(m))
    return weights

def train(X,Y,learning_rate = 0.5, maxEpochs = 100):
    #Modify the input to handle the bias term
    ones = np.ones((X.shape[0],1))
    X = np.hstack((ones,X))
    # Init weights as 0
    weights = np.zeros(X.shape[1]) #n+1 entries
    
    #Iterate over all epochs and make update
    
    for epoch in range(maxEpochs):
         weights  = update(X,Y,weights,learning_rate)            
        
        
         if epoch%10 == 0:
                
                l = loss(X,Y,weights)
                
                print("Epoch %d Loss %.4f"%(epoch,l))
                
                
    return weights
                
    
    
    


# In[10]:


weights = train(X,Y,learning_rate=0.6, maxEpochs = 1000)


# In[11]:


print(weights)


### Perceptron Implementation Part 2



def getPredictions(X_Test, weights, labels=True):
       if X_Test.shape[1] != weights.shape[0]:
       
           ones = np.ones((X_Test.shape[0],1))
           X_Test = np.hstack((ones,X_Test))
           
       probs = predict(X_Test, weights)
       
       if not labels:
           return probs
       else:
           labels = np.zeros(probs.shape)
           labels[probs>=0.5] = 1
       
       return labels
       
   


# In[16]:


x1 = np.linspace(-12.5,5,10)
print(x1)

x2 = -(weights[0] + weights[1]*x1)/weights[2]

print(x2)


# In[17]:


plt.scatter(X[:,0],X[:,1],c=Y, cmap=plt.cm.Accent)
plt.plot(x1,x2,c='red')
plt.show()


# In[15]:


Y_ = getPredictions(X,weights,labels = True)
training_acc = np.sum(Y_==Y)/Y.shape[0]
print(training_acc)




