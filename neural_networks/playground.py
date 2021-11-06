"""
Author: Tassilo Schwarz
Institution: UC Berkeley
Date: Spring 2021
Course: CS189/289A
"""
import numpy as np


def f_relu(z):
    return z if z >= 0 else 0


Z = np.array([[0,4,9],[1,2,3]])

print(Z)

result = np.zeros_like(Z)

d = Z.shape[1]


dY_row = Z[1,:].reshape(1,-1) # as col vector
s = Z[0,:].reshape(1,-1) # the current data point. loop over it


s_is = np.tile(s.reshape(-1,1),(1,d))
s_js = np.tile(s.reshape(1,-1),(d,1))
jacobian = np.exp(s_js-s_is)
np.fill_diagonal(jacobian,np.exp(-s)+1)

s_deriv = np.dot(dY_row,jacobian)

result[0,:] = s_deriv

print("s \n {}".format(s))
print("s_is \n {}".format(s_is))
print("s_js \n {}".format(s_js))
print("jacobian \n {}".format(jacobian))

print("dY_row \n {}".format(dY_row))
print("dY_row times jacobian \n {}".format(s_deriv))

print("result \n {}".format(result))





# print(np.vectorize(f_relu)(A))