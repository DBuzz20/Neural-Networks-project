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



n = len(x[0]) #number of inputs


N = 63
rho = 10 ** (-5)
sigma = 1

#----------------------------------------------------------------

def pred(omega, x, N, sigma):
  v = omega[0:N]
  c = omega[N:]
  C = np.reshape(c, (N, 2))
  X = np.reshape(np.tile(x, N), (len(x), N, 2))
  C1 = np.reshape(np.tile(C, (len(x), 1)), (len(x), N, 2))

  diff = X - C1

  #calcolo norma lungo asse=2 per ottenere la distanza
  dist = np.linalg.norm(diff, axis=2)
  phi = np.exp(-(dist / sigma) ** 2)

  pred = np.dot(phi, v)

  return pred

def err_fun(omega, N, rho, sigma, x, y):
  P=len(x)

  y_pred=pred(omega, x, N, sigma) 
  reg_term=(rho / 2) * (np.linalg.norm(omega)) ** 2

  return (1 / (2 * len(x))) * (np.linalg.norm(y_pred - y.flatten()) ** 2) + reg_term

def grad_err_fun(omega, N, rho, sigma, x, y):
    v = omega[:N]  #N
    c = omega[N:]  #2 * N
    C = c.reshape((N, 2))
    P = len(x)

    diff = x[:,np.newaxis,:] - C[np.newaxis,:,:]
    dist = np.linalg.norm(diff, axis=2)  #(250, N)
    phi = np.exp(-(dist / sigma) ** 2)  #(250, N)
    y_pred = phi@v  #(250,)
    
    #gradiente rispetto a v
    error_term = (y_pred - y.flatten())   #(250,)
    err_v_phi= np.dot(phi.T,error_term) #(1,N)

    dE_v = (1 / P) *err_v_phi   #(N,)

    #gradiente rispetto a c
    error_term_c = error_term[:,np.newaxis] * v[np.newaxis,:] #(250,N)
    aux_c= error_term_c*phi #(250,N)
    aux_c_2=aux_c[:,:,np.newaxis]*diff #(250,N,2)
    dE_c = (1 / P) * (2 / sigma ** 2) * np.sum(aux_c_2, axis=0)  #(N, 2)
    grad = np.concatenate([dE_v, dE_c.flatten()]) + rho * omega  #(3 * N,)
    
    return grad

def plot_function(omega_star, N, sigma):
  x1 = np.linspace(-2, 2)
  x2 = np.linspace(-3, 3)
  
  X, Y = np.meshgrid(x1, x2)

  v = omega_star[0: N]
  c = omega_star[N:]
  C = np.reshape(c, (N, 2))
  
  Z = np.zeros_like(X)  #inizializziamo Z come una matrice della stessa forma di x
  for i in range(N):
    dist_sq = (X - C[i, 0]) ** 2 + (Y - C[i, 1]) ** 2  #calcolo distanza
    gaussian = np.exp(-dist_sq / sigma ** 2)  #calcolo esponenziale
    Z += v[i] * gaussian  # gaussiana pesata
    
  graph = plt.figure(figsize=(10, 10))
  
  ax = graph.add_subplot(111, projection='3d')
  ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
  
  ax.set_xlabel('x')
  ax.set_ylabel('x')
  ax.set_zlabel('z')

  plt.savefig("img_12_AWS.png")

  plt.close()
  
  print("=> The plot has been saved in the current folder\n")


def main(N, rho, sigma,x,y):

  omega0 = np.random.randn(N+2 * N)

  start_t = time.time()
  optimum = minimize(err_fun, omega0, args=(N, rho, sigma,x,y), method='L-BFGS-B',jac=grad_err_fun, options={"ftol": 1e-7, "gtol": 1e-7, "maxfun": 2e3} )
  run_t = time.time() - start_t

  Opt_succ=optimum.success
  Opt_method="L-BFGS-B"
  omega_star=optimum.x
  n_iterations=optimum.nit
  n_obj_fun=optimum.nfev
  n_grad=optimum.njev
  
  starting_error = err_fun(omega0, N, rho, sigma,x,y)
  final_error = err_fun(omega_star, N, rho, sigma,x,y) #=optimum.fun

  gradient_norm_omega_0 = np.linalg.norm(grad_err_fun(omega0, N, rho, sigma,x,y))
  gradient_norm_omega_star = np.linalg.norm(grad_err_fun(omega_star, N, rho, sigma,x,y))
  
  train_err=err_fun(omega_star, N, 0, sigma,x,y) #rho set to 0
  """
  MSE_val= 0
  kf = KFold(n_splits=5, random_state=1895533, shuffle=True)
  for _, val_index in kf.split(x_test):
    x_val_fold = x_test[val_index]
    y_val_fold = y_test[val_index].flatten()

    #COMPUTE MSE for validation set
    Y_val_pred = pred(omega_star,x_val_fold , N, sigma)
    MSE_val += (1 / (2 * len(x_val_fold))) * np.linalg.norm( Y_val_pred - y_val_fold) ** 2

  avg_MSE_val = (MSE_val / kf.get_n_splits())
  """
  validation_error=err_fun(omega_star, N, 0, sigma,x_test,y_test)
  print("------------------------------------------------------------------")
  print("Neurons = %d" %N)
  print("Rho = %f" %rho)
  print("Sigma = %f" %sigma)
  print("Function and gradient tolerance = 1e-7")
  print()
  print("Optimization method = %s" %Opt_method)
  print("Optimization success = %s" %Opt_succ)
  print("Function value at starting point = %f" %starting_error)
  print("Function value in optimum point = %f" %final_error)
  print("Gradient norm at starting point = %f" %gradient_norm_omega_0)
  print("Gradient norm at optimum point = %f" %gradient_norm_omega_star)
  print()
  print("Number of iterations = %d" %n_iterations)
  print("Number of obj. function evaluations = %d" %n_obj_fun)
  print("Number of gradient evaluations = %d" %n_grad)
  print("Time spent to optimize the function = %f" %run_t)
  print()
  print("Training error = %f" %train_err)
  print("Validation error = %f" %validation_error)
  print("------------------------------------------------------------------")
  
  return omega_star

def optimum_pi(params):
  kf = KFold(n_splits=5, random_state=1895533, shuffle=True)

  curr_best_MSE = float("inf")

  avg_MSE_list = []

  for N in params[0]:
    for rho in params[1]:
      for sigma in params[2]:

        print("Current hyperparameters => N:", N, "\tRho:", rho, "\tSigma:", sigma)

        MSE_val= 0
        MSE_train=0

        for train_index, val_index in kf.split(x):
          x_train_fold = x[train_index]
          y_train_fold = y[train_index].flatten()
          x_val_fold = x[val_index]
          y_val_fold = y[val_index].flatten()
          
          omega_star = main(N, rho, sigma,x_train_fold,y_train_fold)
          
          y_hat_train = pred(omega_star, x_train_fold, N,sigma)
          MSE_train += (1 / (2 * len(x_train_fold))) * np.linalg.norm(y_hat_train - y_train_fold) ** 2

          #COMPUTE MSE for validation set
          y_hat_val = pred(omega_star, x_val_fold, N,sigma)
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


params1=[np.arange(10, 201, step=10),np.array([1e-5, 5e-4, 1e-4,5e-3, 1e-3]),np.array([0.1,0.5, 1.0, 1.5,2])]

paramsN=[np.arange(1,202,step=10),np.array([1e-5]),np.array([1.0])]

paramsRHO=[np.array([63]),np.array([1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3]),np.array([1.0])]

paramsSIGMA=[np.array([63]),np.array([1e-5]),np.array([0.1,0.25,0.5,0.75, 1.0,1.25, 1.5,1.75,2,2.5])]

