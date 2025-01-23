from functions_22_AWS import *

np.random.seed(60236)

ris=main(N,rho,sigma)
v_star=ris[0]
C=ris[1]

plot_function(v_star,C, N, sigma)

"""
#OPTIMIZATION OF THE RANDOM SELECTION OF THE CENTERS

multistart() 

#THE BEST SEED FOUND IS: 60236
#with validation error: 0.000821
"""