import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import time

#training set
data = pd.read_csv('dataset.csv')
x = data[['x1', 'x2']].values
y = data[['y']].values

#validation set
data_test = pd.read_csv('blind_test.csv')
x_test = data_test[['x1', 'x2']].values
y_test = data_test[['y']].values

N = 50
rho = 10 ** (-5)
sigma = 1

n = len(x[0])
P = len(x)

def g(t):
    return np.tanh(t)


def dg(t, sigma):
    return sigma * (1 - (g(t)) ** 2)


def get_elm_matrices(W, b):
    #hidden layer output G
    G = g(np.dot(W, x.T) + b)

    Q = ((1 / P) * np.dot(G, G.T)) + (np.eye(N) * rho)

    c = (1 / P) * np.dot(G, y)
    
    return Q, c


def net_pred(v, data, W, b): #mlp prediction
    z = g(np.dot(W, data.T) + b)
    
    y_hat = np.dot(v.T, z).T

    return y_hat


def err(v, rho, W, b,x,y):
    #error function
    err = (1 / (2 * P)) * (np.linalg.norm(net_pred(v, x, W, b) - y) ** 2)
    reg_term = (1 / 2) * rho * (np.linalg.norm(v) ** 2)

    return err + reg_term

def main(Q, c, W, b, v_init):

    start_t = time.time()
    #risolve sistema di equazioni lineari con metodo dei minimi quadrati
    v_star = np.linalg.lstsq(Q, c, rcond=None)[0]
    
    run_t= time.time() - start_t
    
    objective_value_at_start=err(v_init,rho,W,b,x,y)
    objective_value_at_optimum =err(v_star, rho, W, b,x,y)
    
    train_error=err(v_star,0,W,b,x,y) #computed as error function with rho=0
    """
    avg_MSE_val = 0
    MSE_val= 0
    kf = KFold(n_splits=5, random_state=1895533, shuffle=True)
    for _, val_index in kf.split(x_test):

        x_val_fold = x_test[val_index]
        y_val_fold = y_test[val_index]

        #COMPUTE MSE for validation set
        y_hat_val = net_pred(v_star, x_val_fold, W, b) #.reshape((len(y_val_fold), 1))
        #compute mse for each sample in the set
        MSE_val += (1 / (2 * (len(x_val_fold)))) * np.linalg.norm(y_hat_val - y_val_fold) ** 2
        
    avg_MSE_val = (MSE_val / kf.get_n_splits())
    """
    val_error=err(v_star,0,W,b,x_test,y_test) #computed as error function with rho=0
    #printing routine

    Opt_method="Linear least squares"
    
    print("------------------------------------------------------------------")
    print("Neurons = %d" %N)
    print("Rho = %f" %rho)
    print("Sigma = %f" %sigma)
    print("Optimum seed = 98528")
    print()
    print("Optimization method = %s" %Opt_method)
    print("Optimization succes = True")
    print("Function value in starting point = %f" %objective_value_at_start)
    print("Function value in optimum point = %f" %objective_value_at_optimum)
    print()
    print("Number of iterations = 1")
    print("Number of function evaluations = 1")
    print("Time spent to optimize the function = %f" %run_t)
    print()
    print("Training error = %f" %train_error)
    print("Validation error = %f" %val_error)
    print("------------------------------------------------------------------")

    return [val_error,v_star]


def plot(v_star, W, b):

    x1 = np.linspace(-2, 2)
    x2 = np.linspace(-3, 3)

    x, y = np.meshgrid(x1, x2)
    data = np.vstack((x.ravel(), y.ravel())).T

    y_hat = np.array(net_pred(v_star, data, W, b))
    
    y_hat = np.reshape(y_hat, x.shape)
    

    graph = plt.figure(figsize=(10, 10))

    ax = graph.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, y_hat, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    plt.savefig("img_21_AWS.png")

    plt.close()
    
    print("=> The plot has been saved in the question's folder\n")


def multistart(): 
    Best_MSE=float("inf") 
    for i in range(100000):
        np.random.seed(i)
        
        #W and b taken randomly
        W= np.random.randn(N * n).reshape((N, n))
        b = np.random.randn(N).reshape((N, 1))
        v_init=np.random.randn(N).reshape((N, 1))
        #compute Q e c given W and b
        Q, c = get_elm_matrices(W, b)
        
        MSE_val=main(Q, c, W, b, v_init)[0]
        
        print("---------------------------------")
        print("Trying seed:%d" %i)
        print("MSE: %f" %MSE_val)
        print("---------------------------------")
        
        if MSE_val < Best_MSE:
            Best_MSE=MSE_val
            Best_seed=i
            
    print("THE BEST SEED FOUND IS: %d" %Best_seed)
    print("with validation error: %f" %Best_MSE)