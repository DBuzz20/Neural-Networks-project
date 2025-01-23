import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

#CONSTANTS------------------------------------------------------

#DATA RETRIVIAL
#training set
data = pd.read_csv('dataset.csv')
x = data[['x1', 'x2']].values
y = data[['y']].values

#validation set (used in the kfold that computes the validation error)
data_test = pd.read_csv('blind_test.csv')
x_test = data_test[['x1', 'x2']].values
y_test = data_test[['y']].values


n = len(x[0]) 


N = 50
rho = (10 ** (-5))
sigma = 1

#----------------------------------------------------------------

def net_pred(omega, x, N): #network's prediction
  #split the omega into W, bias and v
  W = omega[: N * n].reshape((N, n))
  b = omega[N * n: N * (n + 1)].reshape((N, 1))
  v = omega[N * (n + 1): N * (n + 2)].reshape((N, 1))

  #z = g((W x_transp)+b)
  t= np.dot(W, x.T) + b
  z = g(t)

  #y_hat = (v_transp z)_transp
  y_hat = np.dot(v.T, z).T

  return y_hat

def g(t): #activation function
  return np.tanh(t)


def dg_dt(t, sigma):
  return sigma * (1 - (g(t)) ** 2)

#Error function (regularized training error)
def err_fun(omega, N, rho, sigma, x, y): #passo sigma solo per non avere errori nella minimize ed averlo come parametro per grad_err_fun
  P = len(x)
  err = (1 / (2 * P)) * (np.linalg.norm(net_pred(omega, x, N) - y) ** 2)
  reg_term = (1 / 2) * rho * (np.linalg.norm(omega) ** 2)
  return err + reg_term


def grad_err_fun(omega, N, rho, sigma, x, y):
  P = len(x)
  #split the omega into W, bias and v
  W = omega[: N * n].reshape((N, n))
  b = omega[N * n: N * (n + 1)].reshape((N, 1))
  v = omega[N * (n + 1): N * (n + 2)].reshape((N, 1))

  #compute the z values for all samples and other info
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

  #collect them together
  dE_tot = np.concatenate((dE_dw, dE_db, dE_dv), axis=0)
  error = y_hat - y
  grad = (1 / P) * np.sum(dE_tot * error.T, axis=1) + rho * omega.T

  return grad


def plot_fun(omega_star, N):
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

  plt.savefig("img_11_AWS.png")

  plt.close()

  print("=> The plot has been saved in the current folder\n")


def main(N,rho,sigma,x,y):

  #W ha dim N*n (n inputs), b ha dimensione N, v ha dimensione N (solo un output)
  W = (np.random.randn(N *n))
  b = (np.random.randn(N))
  v = (np.random.randn(N))
  omega0 = np.concatenate((W,b,v), axis=0)

  start_t = time.time()

  optimum = minimize(err_fun, omega0, args=(N, rho, sigma,x ,y), method="L-BFGS-B", jac=grad_err_fun,options={"ftol": 1e-7, "gtol": 1e-7, "maxfun": 2e3})

  run_t = time.time() - start_t

  #Printing routine:
  Opt_succ=optimum.success
  Opt_method="L-BFGS-B"
  omega_star=optimum.x
  n_iterations=optimum.nit
  n_obj_fun=optimum.nfev
  n_grad=optimum.njev
  
  starting_obj_val = err_fun(omega0, N, rho, sigma,x,y)
  final_obj_val = err_fun(omega_star, N, rho, sigma,x,y)
  
  training_error = err_fun(omega_star, N, 0, sigma,x,y) #rho set to 0

  gradient_norm_omega_0 = np.linalg.norm(grad_err_fun(omega0, N, rho , sigma,x,y))
  gradient_norm_omega_star = np.linalg.norm(grad_err_fun(omega_star, N, rho, sigma,x,y))
  """
  avg_MSE_val = 0
  MSE_val= 0
  kf = KFold(n_splits=5, random_state=1895533, shuffle=True)
  for _, val_index in kf.split(x_test):

    x_val_fold = x_test[val_index]
    y_val_fold = y_test[val_index]

    #compute MSE for kfold_validation_set
    y_hat_val = net_pred(omega_star, x_val_fold, N)
    #compute mse for each sample in the kfold_validation_set
    MSE_val += (1 / (2 * (len(x_val_fold)))) * np.linalg.norm(y_hat_val - y_val_fold) ** 2 #error fun with rho=0
  avg_MSE_val = (MSE_val / kf.get_n_splits())
  """
  validation_error=err_fun(omega_star,N,0,sigma,x_test,y_test)
  
  
  print("------------------------------------------------------------------")
  print("Neurons = %d" %N)
  print("Rho = %f" %rho)
  print("Sigma = %f" %sigma)
  print("Function and gradient tollerance = 1e-7")
  print()
  print("Optimization method = %s" %Opt_method)
  print("Optimization success = %s" %Opt_succ)
  print("Function value in starting point = %f" %starting_obj_val)
  print("Function value in optimum point = %f" %final_obj_val)
  print("Gradient norm at starting point = %f" %gradient_norm_omega_0)
  print("Gradient norm at optimum point = %f" %gradient_norm_omega_star)
  print()
  print("Number of iterations = %d" %n_iterations)
  print("Number of function evaluations = %d" %n_obj_fun)
  print("Number of gradient evaluations = %d" %n_grad)
  print("Time spent to optimize the function = %f" %run_t)
  print()
  print("Training error = %f" %training_error)
  print("Validation error = %f" %validation_error)
  print("------------------------------------------------------------------")

  return omega_star

#grid_search

def optimum_pi(params): #"params" is an array [N,rho,sigma]
  kf = KFold(n_splits=5, random_state=1895533, shuffle=True)

  curr_best_MSE = float("inf")
  avg_MSE_list = []

  for N in params[0]:
    for rho in params[1]:
      for sigma in params[2]:

        print("Current hyperparameters => N:", N, "\tRho:", rho, "\tSigma:", sigma)

        MSE_val= 0
        MSE_train= 0

        for train_index,val_index in kf.split(x):
          
          x_train_fold = x[train_index]
          y_train_fold = y[train_index]
          x_val_fold = x[val_index]
          y_val_fold = y[val_index]
          
          omega_star = main(N, rho, sigma,x_train_fold,y_train_fold)
          
          y_hat_train = net_pred(omega_star, x_train_fold, N)
          MSE_train += (1 / (2 * len(x_train_fold))) * np.linalg.norm(y_hat_train - y_train_fold) ** 2

          #COMPUTE MSE for validation set
          y_hat_val = net_pred(omega_star, x_val_fold, N)
          MSE_val += (1 / (2 * len(x_val_fold))) * np.linalg.norm(y_hat_val - y_val_fold) ** 2
          
        avg_MSE_train = (MSE_train / kf.get_n_splits())
        avg_MSE_val = (MSE_val / kf.get_n_splits())

        avg_MSE_list.append([avg_MSE_train, avg_MSE_val])

        if avg_MSE_val < curr_best_MSE:
          curr_best_MSE = avg_MSE_val
          print("BETTER PARAMS FOUND:")
          print("Neurons = %d" %N)
          print("Sigma = %f" %sigma)
          print("Rho = %f" %rho)
          print("Validation_error = %f" %curr_best_MSE)
          best_params = [N, rho, sigma]
          
  print("List of the average MSE = ", avg_MSE_list)
  print(best_params)
  print(curr_best_MSE)

  return 

#--------------------------------------------------------------------------------------------
#parametri=[N,rho,sigma]
parametri=[np.arange(1, 122, step=2),np.array([1e-5, 7.5e-4, 5e-4, 2.5e-4, 1e-4, 7.5e-3, 5e-3, 2.5e-3, 1e-3]),np.arange(0.1, 2.50, step=0.25)]

#----------------------------------
paramsN2=[np.arange(40, 70, step=1),np.array([1e-5]),np.array([1.0])]


#----------------------------------
paramsN=[np.arange(1, 122, step=1),np.array([1e-5]),np.array([1.0])]

#--------------------------
paramsRHO=[np.array([50]),np.array([1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3]),np.array([1.0])]

#--------------------------
paramsSIGMA=[np.array([50]),np.array([1e-5]),np.array([0.1,0.25,0.5,0.75, 1.0,1.25, 1.5,1.75,2,2.5])]