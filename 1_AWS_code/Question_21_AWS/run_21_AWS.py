from functions_21_AWS import *

np.random.seed(98528)

#W and b taken randomly
W= np.random.randn(N * n).reshape((N, n))
b = np.random.randn(N).reshape((N, 1))
v_init=np.random.randn(N).reshape((N, 1))

#compute Q e c, given W and b
Q, c = get_elm_matrices(W, b)

v_star = main(Q, c, W, b, v_init)[1]

plot(v_star, W, b)

"""
multistart()
OUTPUT: 
THE BEST SEED FOUND IS: 98528
with validation error: 0.001226
"""