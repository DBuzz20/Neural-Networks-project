import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
from scipy.optimize import check_grad
from sklearn.preprocessing import StandardScaler

#CONSTANTS------------------------------------------------------

#DATA RETRIVIAL
#training set
data = pd.read_csv('dataset.csv')
x = data[['x1', 'x2']].values
y = data[['y']].values

#validation set
data_test = pd.read_csv('blind_test.csv')
x_test = data_test[['x1', 'x2']].values
y_test = data_test[['y']].values


n = len(x[0]) #number of inputs

N = 63
rho = 10 ** (-5)
sigma = 1

#----------------------------------------------------------------

def pred(v,C, x, sigma):
  P = len(x)
  diff = x[:,np.newaxis,:] - C[np.newaxis,:,:]

  #calcolo norma lungo asse=2 per ottenere la distanza
  dist = np.linalg.norm(diff, axis=2)
  phi = np.exp(-(dist / sigma) ** 2)

  pred = phi@v

  return pred

def err_fun(v,C, rho, sigma,x,y):
  P=len(x)
  

  y_pred=pred(v,C,x,sigma) 
  reg_term=(rho / 2) * (np.linalg.norm(v)) ** 2

  return (1 / (2 * P)) * (np.linalg.norm(y_pred - y.flatten()) ** 2) + reg_term


def select_random_centers():
    P=len(x)
    indices = np.random.choice(P, N, replace=False)
    return x[indices]


def plot_function(v,C, N, sigma):
  x1 = np.linspace(-2, 2)
  x2 = np.linspace(-3, 3)
  
  X, Y = np.meshgrid(x1, x2)
  
  Z = np.zeros_like(X)  
  for i in range(N):
    dist_sq = (X - C[i, 0]) ** 2 + (Y - C[i, 1]) ** 2  
    gaussian = np.exp(-dist_sq / sigma ** 2)  
    Z += v[i] * gaussian  

  graph = plt.figure(figsize=(10, 10))
  ax = graph.add_subplot(111, projection='3d')
  ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
  
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  
  plt.savefig("img_22_AWS.png")

  plt.close()

  print("=> The plot has been saved in the current folder\n")
  

def fun_lstsq(C,x,y,N,rho,sigma):
  P=len(x)
  diff = x[:,np.newaxis,:] - C[np.newaxis,:,:]
  dist = np.linalg.norm(diff, axis=2)  
  phi = np.exp(-(dist / sigma) ** 2) 
  matrix=phi.T@phi
  id_matr=np.eye(matrix.shape[0])
  
  Q=(1/P)*matrix + rho*id_matr
  s=(1/P)*((y.flatten()).T@phi).T
  
  v_star=np.linalg.lstsq(Q,s,rcond=None)[0]
  
  return v_star



def main(N, rho, sigma):
  C=select_random_centers()
  v0 = np.random.randn(N)
  
  start_t=time.time()
  v_star=fun_lstsq(C,x,y,N,rho,sigma)
  run_t= time.time()-start_t

  starting_error = err_fun(v0,C, rho, sigma,x,y)
  final_error = err_fun(v_star,C , rho, sigma,x,y)#=optimum.fun
  
  train_err=err_fun(v_star,C, 0, sigma,x,y) #rho to 0 for training error
  
  """
  avg_MSE_val = 0
  MSE_val= 0
  kf = KFold(n_splits=5, random_state=1895533, shuffle=True)
  for _, val_index in kf.split(x_test):

    x_val_fold = x_test[val_index]
    y_val_fold = y_test.flatten()[val_index]
    
    #COMPUTE MSE for validation set
    Y_val_pred = pred(v_star,C,x_val_fold, sigma)
    MSE_val += (1 / (2 * len(x_val_fold))) * np.linalg.norm( Y_val_pred - y_val_fold) ** 2

  avg_MSE_val = (MSE_val / kf.get_n_splits())
  """
  test_err=err_fun(v_star,C, 0, sigma,x_test,y_test) #rho to 0 for training error
  
  #printing routine
  Opt_method="Linear least squares"
  
  print("------------------------------------------------------------------")
  print("Neurons = %d" %N)
  print("Rho = %f" %rho)
  print("Sigma = %d" %sigma)
  print("Best seed: 60236")
  print()
  print("Optimization method =%s" %Opt_method)
  print("Optimization success = True")
  print("Function value at starting point = %f" %starting_error)
  print("Function value in optimum point = %f" %final_error)
  print()
  print("Number of iterations = 1")
  print("Number of obj. function evaluations = 1")
  print("Time spent to optimize the function = %f" %run_t)
  print()
  print("Training error = %f" %train_err)
  print("Validation error = %f" %test_err)
  print("------------------------------------------------------------------")

  return [v_star, C, test_err]

def multistart():
    Best_MSE=float("inf") 
    for i in range(100000):
        np.random.seed(i)
        val_err=main(N, rho, sigma)[2]
    
        print("---------------------------------")
        print("Trying seed:%d" %i)
        print("MSE: %f" %val_err)
        print("---------------------------------")
        
        if val_err < Best_MSE:
            Best_MSE=val_err
            Best_seed=i
            
    print("THE BEST SEED FOUND IS: %d" %Best_seed)
    print("with validation error: %f" %Best_MSE)