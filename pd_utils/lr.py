import numpy as np

# FUNCTION: sigmoid

def sigmoid(z):
    a=1/(1+np.exp(-z))
    return a


# FUNCTION: propagate

def propagate(w, b, X, Y):    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T,X)+b)                                    # compute activation
    cost = (-1/m)*(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))           # compute cost
       
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1/m)*(np.dot(X,(A-Y).T))
    db = (1/m)*np.sum(A-Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost



# FUNCTION: optimize -This function optimizes w and b by running a gradient descent algorithm

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation 
        grads, cost = propagate(w,b,X,Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule 
        w = w-(learning_rate*dw)
        b = b-(learning_rate*db)
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs



# FUNCTION: predict_lr -Predicts whether the label is 0 or 1 using learned logistic regression parameters (w, b)

def predict_lr(w, b, X):
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture    
    A = sigmoid(np.dot(w.T,X)+b)

    for i in range(A.shape[1]):        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        Y_prediction[0,i]=0 if A[0,i]<=0.5 else 1
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction


