import numpy as np
import matplotlib as mpl
import pdb

mpl.use("TkAgg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/mnt/d/Studium_NIM/work/Codes/MOSAiC/")


class geom_obj:

	def __init__(self, name, extent):

		if name == 'circle':
			self.size = extent
			self.x = np.linspace(-self.size, self.size, 1000)
			self.y = np.sqrt(self.size**2 - self.x**2)
			self.centre = [0.0, 0.0]
			self.distance = np.sqrt((self.x-self.centre[0])**2 + (self.y-self.centre[1])**2)
			self.n_data = len(self.distance)

		elif name == 'square':
			self.size = extent
			self.x = np.array([np.linspace(-self.size, self.size, 1000), 
								np.ones((1000,))*self.size,
								np.linspace(self.size, -self.size, 1000),
								np.ones((1000,))*(-self.size)])
			self.y = np.array([0.0*self.x[0,:] + self.size,
								np.linspace(self.size, -self.size, 1000),
								0.0*self.x[2,:] - self.size,
								np.linspace(-self.size, self.size, 1000)])
			self.x = self.x.flatten()
			self.y = self.y.flatten()
			self.centre = [0.0, 0.0]
			self.distance = np.sqrt((self.x-self.centre[0])**2 + (self.y-self.centre[1])**2)
			self.n_data = len(self.distance)


# define an object:
radius = 1.0
edges = 1.0
circle = geom_obj('circle', radius)
square = geom_obj('square', edges)



# visualize object and histogram of distances
f1, (a1, a2) = plt.subplots(1,2)
f1.set_size_inches((12,6))

axlims = [-1.5*radius, 1.5*radius]

a1.plot(square.x, square.y, color=(0,0,0), linewidth=1.2)
# a1.plot(circle.x, -circle.y, color=(0,0,0), linewidth=1.2)

a1.plot([axlims[0], axlims[1]], [0, 0], color=(0,0,0), linewidth=0.75)
a1.plot([0, 0], [axlims[0], axlims[1]], color=(0,0,0), linewidth=0.75)

# histogram:
bins = np.linspace(0.0, 2.0*square.size, 32)
weights_data = np.ones((square.n_data,)) / float(square.n_data)
a2.hist(square.distance, bins=bins, weights=weights_data, color=(0.8,0.8,0.8), ec=(0,0,0))

a1.minorticks_on()
a1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

a2.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

a1.set_xlim(axlims[0], axlims[1])
a1.set_ylim(axlims[0], axlims[1])

a1.set_aspect('equal')


plt.show()

