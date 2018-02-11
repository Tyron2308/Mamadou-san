import numpy as np


import numpy as np
from scipy import linalg
from scipy import signal
x = np.array([0,0,1,0,0,2,0,0,0]) # 9
h = np.array([0,1,2,0]) # 4
y = signal.convolve(x, h, mode='same')
print "x", x
print "h", h
print "y(conv):", y

padding = np.zeros(len(x)-1, h.dtype)
first_col = np.r_[h, padding]
first_row = np.r_[h[0], padding]
H = linalg.toeplitz(first_col, first_row)[1:len(x)+1,:]
print "shape", H.shape, x.shape
y = np.sum(np.multiply(x,H), 1)
print "y(mult):", y
print "**********************"
x = np.array([0,0,1,0,0,2,0,0,0]) # nsamp
x = np.tile(x,[10,1]) # n_ex x n_samp
h = np.array([0,1,2,0]) # n_samp
h = np.tile(h,[10,1]) # n_ex x n_samp
y = np.zeros([x.shape[0], x.shape[1]])
for i in range(0,x.shape[0]):
    y[i,:] = signal.convolve(x[i,:], h[i,:], mode='same')
print "x", x
print "h", h
print "y(conv):", y



# # set up the toeplitz matrix
# H = np.zeros([ x.shape[0], x.shape[1], x.shape[1] ]) # n_ex x n_samp x n_samp
# for i in range(0,x.shape[0]):
#     padding = np.zeros(x.shape[1]-1, h.dtype) #
#     first_col = np.r_[h[i,:], padding] #
#     first_row = np.r_[h[i,0], padding] #
#     H[i,:,:] = linalg.toeplitz(first_col, first_row)[1:x.shape[1]+1,:]
# print "H shape", H.shape
# print H[0,:,:]
# x = x.reshape([x.shape[0], 1, x.shape[1]])
# x = np.tile(x, [1,x.shape[1],1])
# y = np.sum(np.multiply(x,H), 2)
# print "y(mult):", y
# print "**********************"
# h = np.array([0,1,2,3,4,5,6,7,8], dtype='int32')
# padding = np.zeros(len(x)-1, h.dtype)
# first_col = np.r_[h, padding]
# first_row = np.r_[h[0], padding]
# H = linalg.toeplitz(first_col, first_row)[1:len(x)+1,:]
# print H
#
#
# Posted in Uncategorized
# Post navigation
