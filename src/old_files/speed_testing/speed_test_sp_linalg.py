import time
import numpy as np
import scipy as sp
import scipy.sparse as sp_sparse
import scipy.sparse.linalg as sp_sparse_linalg
import scipy.linalg as sp_linalg

N = 2000

np.random.seed(2)
M = sp_sparse.rand(N, N, format="csc", density=.1)
Mn = M.toarray() #numpy version

c = sp_sparse.rand(N, 1, format="csc")
cn = c.toarray()

#t0 = time.time()
#I = sp_sparse_linalg.inv(M)
#x = I.dot(c)
#print("19: "+str(x[0]))

t1 = time.time()
x = sp_sparse_linalg.spsolve(M, c)
print("23: "+str(x[0]))

t2 = time.time()
(_, lnDM) = np.linalg.slogdet(Mn)
Mc = Mn
Mc[:,0] = np.transpose(cn)
(_, lnDMc) = np.linalg.slogdet(Mc)
x0 = np.exp( lnDMc - lnDM )
print("32: "+str(x0))

# Neither BICG nor CG are converging for some reason
t3 = time.time()
x, code = sp_sparse_linalg.bicg(M, c.toarray())
print("38: "+str(code)+": "+str(x[0]))

t4 = time.time()
x, code = sp_sparse_linalg.cg(M, c.toarray())
print("40: "+str(code)+": "+str(x[0]))

t5 = time.time()


#ti = t1 - t0
ts = t2 - t1
tc = t3 - t2
tbcg = t4 - t3
tcg = t5 - t4
print(f"N = {N}")
#print(f"Inversion time: {ti}")
print(f"spsolve time: {ts}")
print(f"Cramer's rule time: {tc}")
print(f"Bicongugate Gradiant time: {tbcg}")
print(f"Congugate Gradiant time: {tcg}")
print(f"C / S: {tc / ts}")
print(f"C / BCG: {tc / tbcg}")
print(f"C / CG: {tc / tcg}")


