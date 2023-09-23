#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import matplotlib.pyplot as plt


# In[32]:


def generate_examples(num = 1000):
    W = [1.0, -3.0]
    b = 1.0
    
    W = np.reshape(W, (2, 1))
    X = np.random.rand(num, 2)
    
    y = b + np.dot(X, W)
    y = np.reshape(y, (num, 1))
    
    return X, y


# In[33]:


class Model:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.random.rand(num_features, 1)
        self.b = np.random.rand()

    def forward_pass(self, X):
        y_hat = self.b + np.dot(X, self.W)
        return y_hat

    def compute_loss(self, y_hat, y_true):
        return np.sum(np.square(y_hat - y_true)) / (2 * y_hat.shape[0])
    def backward_pass(self, X, y_true, y_hat):
        m = y_true.shape[0]
        db = (1/m) * np.sum(y_hat - y_true)
        dW = (1/m) * np.dot((y_hat - y_true).T, X)
        return dW.T, db

    def update_params(self, dW, db, lr):
        self.W = self.W - lr * dW
        self.b = self.b - lr * db

    def train(self, x_train, y_train, iterations, lr):
        losses = []
        for i in range(iterations):
            y_hat = self.forward_pass(x_train)
            loss = self.compute_loss(y_hat, y_train)
            losses.append(loss)
            dW, db = self.backward_pass(x_train, y_train, y_hat)
            self.update_params(dW, db, lr)
            if i % int(iterations / 10) == 0:
                print('iter: {}, loss: {: .4f}' .format(i, loss))
        return losses


# In[34]:


model = Model(2)


# In[35]:


X_train, y_train = generate_examples()


# In[36]:


losses = model.train(X_train, y_train, 1000, 3e-3)


# In[37]:


plt.plot(losses);


# In[38]:


model_untrained = Model(2)
x_test, y_test = generate_examples(500)
print(x_test.shape, y_test.shape)


# In[39]:


preds_untrained = model_untrained.forward_pass(x_test)
preds_trained = model.forward_pass(x_test)


# In[40]:


plt.figure(figsize = (5, 5))
plt.plot(preds_untrained, y_test, 'rx', label = 'untrained')
plt.plot(preds_trained, y_test, 'b.', label = 'trained')
plt.legend()
plt.xlabel('predictions')
plt.ylabel('ground truth')
plt.show()


# In[ ]:




