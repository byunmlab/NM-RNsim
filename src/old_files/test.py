from ResistorNetwork import ResistorNetwork as RN
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import time



# Test different Resistance calculations
t0 = time.time()
rn = RN(120, rand_s=7)
t1 = time.time()
print(f"Building RN: {t1-t0:.3f}s")

print( "R from 0 to 10: ", rn.R_ij(0,10) )
t2 = time.time()
print(f"R_ij: {t2-t1:.3f}s")

#print( "R from 0 to 10: ", rn.R_ij_full(0,10) )
print( "R from 0 to 10: ", rn.R_ij(0,10, lsqr=True) )
t3 = time.time()
print(f"R_ij(lsqr): {t3-t2:.3f}s")
#print(f"R_ij_full: {t3-t2:.3f}s")

#print( "R from 0 to 10: ", rn.R_ij_inv(0,10) )
t4 = time.time()
#print(f"R_ij_inv: {t4-t3:.3f}s")

print( "R from 0 to 10: ", rn.R_ij_det(0,10) )
t5 = time.time()
print(f"R_ij_det: {t5-t4:.3f}s")

rn.apply_v(5, 0, 10)
fig, ax = rn.draw(color_attrib="v")
fig.show()

input()
quit()


M = sps.random(6,6, .75, "csr")#.tolil()
print("M: ")
print(M)
print("M[:, 2] = ", M[:, 2])
Mp = sps.lil_matrix((6,5))
Mp[:,0:2] = M[:,0:2]
Mp[:,2:5] = M[:,3:6]
print("Mp: ")
print(Mp)



quit()




# Test removing nodes and index_to_key & key_to_index
print("Initial Nodes: ", rn.get())

for ni in range(5,10):
  rn.remove(ni)
  
print("Nodes after deleting: ", rn.get())

print("node at index 8 has the name: ", rn.index_to_key(8))
print("node '15' is at index: ", rn.key_to_index(15))

fig, ax = rn.draw()
fig.show()
input("ENTER")
