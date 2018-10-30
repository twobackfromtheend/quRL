
# coding: utf-8

# In[6]:


from qutip import *
import numpy as np


# In[98]:


t = np.linspace(0,200,10)

#H0 = sigmaz()
def H1coeff(t,*args):
    return -0.01*t

hx = H1coeff(t)

Hz = sigmaz()

H0 = -sigmaz()
H1 = -sigmax()
H = [H0, [H1,hx]]




psi0 = basis(2,0)
psix = (basis(2,0) + basis(2,1)).unit()

#t = np.linspace(0,2*np.pi,100)

result = mesolve(H, psi0, t, [], [])

x=[]
y=[]
z=[]

for i in result.states:
    x.append([expect(sigmax(), i)])
    y.append([expect(sigmay(), i)])
    z.append([expect(sigmaz(), i)])


#result.states

# why does it not evaluate expectation of sigmaz() for times t


#from pylab import*
#import matplotlib.animation as animation
#from mpl_toolkits.mplot3d import Axes3D
#
#
#fig = figure()
#ax = Axes3D(fig,azim=-40,elev=30)
#sphere = Bloch(axes=ax)
#
#def animate(i):
#    sphere.clear()
#    sphere.add_vectors(result.states)
#    sphere.add_points([x,y,z])
#    sphere.make_sphere()
#    return ax
#
#def init():
#    sphere.vector_color = ['r']
#    return ax
#
#ani = animation.FuncAnimation(fig, animate, np.arange(len(result.states)),
#init_func=init, blit= True, repeat=True)


b=Bloch()
b.add_points([x,y,z])
b.add_states(result.states)
b.vector_color = 'b'
b.make_sphere()
b.show()
# b.clear()






