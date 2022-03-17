import time
import numpy as np
#import scipy as sp
#from scipy.sparse import random as sci_sp_rnd

N = 2000

np.random.seed(2)
M = np.random.rand(N, N)
#M = sci_sp_rnd(N, N)

c = np.random.rand(N, 1)

t0 = time.time()
I = np.linalg.inv(M)
x = np.dot(I, c)
print(x[0])

t1 = time.time()
x = np.linalg.solve(M, c)
print(x[0])

t2 = time.time()
(_, lnDM) = np.linalg.slogdet(M)
Mc = M
Mc[:,0] = np.transpose(c)
(_, lnDMc) = np.linalg.slogdet(Mc)
x0 = np.exp( lnDMc - lnDM )
print(x0)
t3 = time.time()


ti = t1 - t0
ts = t2 - t1
tc = t3 - t2
print(f"N = {N}")
print(f"Inversion time: {ti}")
print(f"Solving time: {ts}")
print(f"Cramer's rule time: {tc}")
print(f"I / S: {ti / ts}")
print(f"I / C: {ti / tc}")


