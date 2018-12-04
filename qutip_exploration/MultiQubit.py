from qutip import *
import numpy as np
import matplotlib.pyplot as plt


def integrate(L: int, h: float, psi0: Qobj, tlist: list):
    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    sx_list = []
    sy_list = []
    sz_list = []
    exp = []

    for n in range(L):
        op_list = []
        for m in range(L):
            op_list.append(si)
            # print ('hi', op_list)

        op_list[n] = sx  # Replaces the nth element of identity matrix list with the Sx matrix
        # print ('wtf,' , op_list, 'wtf')
        sx_list.append(tensor(op_list))
        # Resulting tensor operates on the nth qubit only.
        # sx_list contains the n sx that operate on the n qubits, with each index operating on a certain qubit

        op_list[n] = sy
        sy_list.append(tensor(op_list))

        op_list[n] = sz
        sz_list.append(tensor(op_list))

    exp.append(sx_list)
    exp.append(sy_list)
    exp.append(sz_list)

    # print (sx_list)

    H = 0

    for n in range(L):
        H += -sz_list[n]
        H += -h * sx_list[n]

        if n < L - 1:
            # print (n)
            H += -sz_list[n + 1] * sz_list[n]

        else:
            # a = -sz_list[n] * sz_list[0]
            H += -sz_list[n] * sz_list[0]
            # print(a)



    explist = []
    for i in range(3):
        result = mesolve(H, psi0, tlist, c_ops=[], e_ops=exp[i])
        explist.append(result.expect)

    return H
    # result = mesolve(H, psi0, tlist, c_ops=[], e_ops = [])
    # return result.states


L = 3
h = 2

psi_list = [basis(2, 0)]

for n in range(L - 1):
    psi_list.append(basis(2, 1))

psi0 = tensor(psi_list)

tlist = np.linspace(0, 50, 200)

sz_expt = integrate(L, h, psi0, tlist)
# print(psi0, sz_expt*psi0)

x = sz_expt.groundstate()[1]
y = (-sigmaz() -2*sigmax()).groundstate()
print(x, sz_expt*x)



# print((sz_expt[0])[0])
# print((sz_expt[1])[0])
# print((sz_expt[2])[0])

# rmag1 = []
# rmag2 = []
# rmag3 = []
#
# for i in range(len((sz_expt[0])[0])):
#     r2 = (((sz_expt[0])[0])[i]) ** 2 + (((sz_expt[1])[0])[i]) ** 2 + (((sz_expt[2])[0])[i]) ** 2
#     rmag1.append(r2)
#
# for i in range(len((sz_expt[0])[0])):
#     r2 = (((sz_expt[0])[1])[i]) ** 2 + (((sz_expt[1])[1])[i]) ** 2 + (((sz_expt[2])[1])[i]) ** 2
#     rmag2.append(r2)
#
# for i in range(len((sz_expt[0])[0])):
#     r2 = (((sz_expt[0])[2])[i]) ** 2 + (((sz_expt[1])[2])[i]) ** 2 + (((sz_expt[2])[2])[i]) ** 2
#     rmag3.append(r2)
#
# fig, ax = plt.subplots(figsize=(10, 6))
# b = Bloch()
#
# for n in range(L):
#     ax.plot(tlist, ((sz_expt[2])[n]), label=r'$\langle\sigma_z^{(%d)}\rangle$' % n)
#
# for i in range(L):
#     b.add_points([(sz_expt[0])[i], (sz_expt[1])[i], (sz_expt[2])[i]])
#
# ax.legend(loc=0)
# ax.set_xlabel(r'Time [ns]')
# ax.set_ylabel(r'\langle\sigma_z\rangle')
# ax.set_title(r'Dynamics of a spin chain')
# plt.show(), b.show()

# rs = [rmag1, rmag2, rmag3]
# plt.plot(tlist, rmag1)
# plt.plot(tlist, rmag2)
# plt.plot(tlist, rmag3)
# plt.show()

