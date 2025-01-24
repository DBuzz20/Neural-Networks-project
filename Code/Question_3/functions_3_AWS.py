import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.optimize import minimize
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


def dg_dt(t, sigma):
    return sigma * (1 - (g(t)) ** 2)


def net_pred(omega, data, N): #network's prediction
  #split omega into W, bias and v
  W = omega[: N * n].reshape((N, n))
  b = omega[N * n: N * (n + 1)].reshape((N, 1))
  v = omega[N * (n + 1): N * (n + 2)].reshape((N, 1))

  #z = g((W x_transp)+b)
  t= np.dot(W, data.T) + b
  z = g(t)

  #y_hat = (v_transp z)_transp
  y_hat = np.dot(v.T, z).T

  return y_hat


def err_fun(omega, N, rho,x,y): #generalization error function
    err = (1 / (2 * P)) * (np.linalg.norm(net_pred(omega, x, N) - y) ** 2)
    reg_term = (1 / 2) * rho * (np.linalg.norm(omega) ** 2)
    return err + reg_term

def grad_err_fun(omega, N, rho, sigma):
    #split the omega into W, bias and v
    W = omega[: N * n].reshape((N, n))
    b = omega[N * n: N * (n + 1)].reshape((N, 1))
    v = omega[N * (n + 1): N * (n + 2)].reshape((N, 1))

    t= np.dot(W, x.T) + b
    z = g(t)
    y_hat = np.dot(v.T, z).T
    dz = dg_dt(t, sigma)

    #derivative respect to W
    dE_dw = (np.repeat((v * dz), repeats=n, axis=0)*np.tile(x.T, (N, 1)) )

    #derivative respect to b
    dE_db = (v * dz)

    #derivative respect to v
    dE_dv = z

    dE_tot = np.concatenate((dE_dw, dE_db, dE_dv), axis=0)
    error = y_hat - y
    grad = (1 / P) * np.sum(dE_tot * error.T, axis=1) + rho * omega.T

    return grad


def err_fun_Wb(W_b, N, rho, v): #non-convex error function
    omega = np.concatenate((W_b.reshape(N * (n + 1), 1), v), axis=0)

    err = (1 / (2 * P)) * (np.linalg.norm(net_pred(omega, x, N) - y) ** 2)
    reg_term = (1 / 2) * rho * (np.linalg.norm(W_b) ** 2)
    return err + reg_term


def grad_err_fun_Wb(W_b, N, rho, v): #non-convex function's gradient
    W = W_b[: N * n].reshape((N, n))
    b = W_b[N * n: N * (n + 1)].reshape((N, 1))

    t=np.dot(W, x.T) + b
    z = g(t)
    y_hat= np.dot(v.T, z).T
    dz = dg_dt(np.dot(W, x.T) + b, sigma)

    #derivative wrt W
    dE_dw = np.repeat((v * dz), repeats=n, axis=0) * np.tile(x.T, (N, 1))

    #derivative wrt b
    dE_db = v * dz
    
    dE_tot = np.concatenate((dE_dw, dE_db), axis=0)
    error = y_hat - y
    grad = (1 / P) * np.sum(dE_tot * error.T, axis=1) + rho * W_b.T

    return grad.reshape(-1)

#convex optimization
def optimize_v(N, rho, sigma, W, b):
    # compute W and c
    G = g(np.dot(W, x.T) + b)
    
    Q = ((1 / P) * np.dot(G, G.T)) + (np.identity(N) * rho)
    c = (1 / P) * np.dot(y.T, G.T).T

    #least squares method optimization
    v_star = np.linalg.lstsq(Q, c, rcond=None)[0]
    return v_star


def main():
    #parameters:
    nfev, njev, n_it= 0, 0, 0
    max_iter = 150

    maxPatience = 20
    pat_counter=0

    tol_init=1e-6
    grad_toll=1e-6
    fun_toll=1e-6
    tol_reduction_factor = 0.9  #reduction term (10%)
    it_interval = 10  #reduce tollerance after each 10 iterations
    
    #matrix initialization
    W = np.random.randn(N, n)
    b = np.random.randn(N, 1)
    v_star = np.random.randn(N, 1)

    #starting omega
    omega = np.concatenate([W.flatten(), b.flatten(), v_star.flatten()])
    omega0=omega

    W_b = np.concatenate((W.reshape((N * n, 1)), b))

    best_error=float("inf")

    start_t = time.time()

    while (np.linalg.norm(grad_err_fun(omega, N, rho, v_star)) < grad_toll or n_it < max_iter):
        #convex optimization with v
        v_star = optimize_v(N, rho, sigma, W, b).reshape((N, 1))
        nfev += 1
        
        W_b = np.concatenate((W.reshape((N * n, 1)), b))
        #non convex optimization with W and b
        opt = minimize(err_fun_Wb, W_b.flatten(), args=(N, rho, v_star), method="L-BFGS-B", jac=grad_err_fun_Wb, options={"gtol": grad_toll, "ftol": fun_toll, "maxfun": 2e3})
        nfev += opt.nfev
        njev += opt.njev
        W_b = opt.x
        W = W_b[: N * n].reshape((N, n))
        b = W_b[N * n: N * (n + 1)].reshape((N, 1))
        
        omega = np.concatenate((W_b.reshape(N * (n + 1), 1), v_star), axis=0)
        curr_error = err_fun(omega, N, rho,x,y)
        nfev += 1
        
        #dynamic tollerance
        if n_it % it_interval == 0 and n_it > 0:
            grad_toll *= tol_reduction_factor
            fun_toll *= tol_reduction_factor

        #if i get a better error, reset patience. Otherwise, update the patience counter
        if best_error > curr_error:
            pat_counter = 0
            best_error = curr_error   
        else:
            pat_counter += 1
        
        #if i dont get a better error for 'maxPatience' times, stop the cycle
        if pat_counter > maxPatience: 
            break

        #stop if it takes too much time 
        if time.time() - start_t > 9:
            #print("TIMEOUT") 
            break
        n_it += 1

    run_t = time.time() - start_t

    starting_err_fun=err_fun(omega0,N,rho,x,y)
    final_err_fun=err_fun(omega,N,rho,x,y)
    
    training_error=err_fun(omega,N,0,x,y) #rho=0 to compute training error
    """
    MSE_val= 0
    kf=KFold(n_splits=5, random_state=1895533, shuffle=True)
    for _, val_index in kf.split(x_test):

        x_val_fold = x_test[val_index]
        y_val_fold = y_test[val_index]        

        y_hat_val = net_pred(omega, x_val_fold, N)
        MSE_val += (1 / (2 * (len(x_val_fold)))) * np.linalg.norm(y_hat_val - y_val_fold) ** 2

    avg_MSE_val = (MSE_val / kf.get_n_splits())
    """
    test_error=err_fun(omega,N,0,x_test,y_test)
    
    #printing routine
    Opt_method_Wb="L-BFGS-B"
    Opt_method_v="LSTSQ (Numpy least square method)"

    print("------------------------------------------------------------------")
    print("Neurons = %d" %N)
    print("Rho = %f" %rho)
    print("Sigma = %f" %sigma)
    print("Maximum number of cycle iterations = %d" %max_iter)
    print("Maximum patience parameter = %d" %maxPatience)
    print("Initial gradient tolerance = %.8f" %tol_init)
    print("Tolerance reduced by 10","%","each %d iterations" %it_interval)
    print("Last gradient tolerance = %.8f" %grad_toll)
    print()
    print("Optimization method (for input weights) = %s" %Opt_method_Wb)
    print("Optimization method (for output weights) = %s" %Opt_method_v)
    print("Function value in starting point = %f" %starting_err_fun)
    print("Function value in optimum point = %f" %final_err_fun)
    print()
    print("Number of cycle iterations = %d" %n_it)
    print("Number of function evaluations = %d" %nfev)
    print("Number of gradient evaluations = %d" %njev)
    print("Time spent to optimize the function = %f" %run_t)
    print()
    print("Training error = %f" %training_error)
    print("Validation error = %f" %test_error)
    print("------------------------------------------------------------------")

    return omega


def plot(omega_star, N):
    x1 = np.linspace(-2, 2)
    x2 = np.linspace(-3, 3)

    x, y = np.meshgrid(x1, x2)
    data = np.vstack((x.ravel(), y.ravel())).T

    y_hat = np.array(net_pred(omega_star, data, N))
    y_hat = np.reshape(y_hat, x.shape)

    graph = plt.figure(figsize=(10, 10))

    ax = graph.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, y_hat, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.savefig("img_3_AWS.png")
    plt.close()
    print("=> The plot has been saved in the current folder\n")
    
