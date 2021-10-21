"""
===========================
More triangular 3D surfaces
===========================

Two additional examples of plotting surfaces with triangular mesh.

The first demonstrates use of plot_trisurf's triangles argument, and the
second sets a Triangulation object's mask and passes the object directly
to plot_trisurf.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
#import sympy as sp


fig = plt.figure(figsize=plt.figaspect(0.5))

#============
# First plot
#============

N = 8

# Make a mesh in the space of parameterisation variables u and v
u = np.linspace(0, 1, endpoint=True, num=2)
v = np.arange(start=0, stop= N / 3 * np.pi + 0.001, step=np.pi/3)
u, v = np.meshgrid(u, v)
u, v = u.flatten(), v.flatten()
#sp.pprint(sp.Matrix(np.vstack((u, v))).T)

# This is the Mobius mapping, taking a u, v pair and returning an x, y, z
# triple
x = u * np.cos(v)
y = u * np.sin(v)
z = v

# Triangulate parameter space to determine the triangles
t = np.array([[0, 1, 3], [0, 3, 2]])
T = np.vstack([t + 2 * i for i in range(0, N)])

tri = mtri.Triangulation(u, v, triangles=T)

# Plot the surface.  The triangles in parameter space determine which x, y, z
# points are connected by an edge.
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.Spectral)

#ax.set_zlim(-1, 1)


##============
## Second plot
##============
#
## Make parameter spaces radii and angles.
#n_angles = 36
#n_radii = 8
#min_radius = 0.25
#radii = np.linspace(min_radius, 0.95, n_radii)
#
#angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
#angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
#angles[:, 1::2] += np.pi/n_angles
#
## Map radius, angle pairs to x, y, z points.
#x = (radii*np.cos(angles)).flatten()
#y = (radii*np.sin(angles)).flatten()
#z = (np.cos(radii)*np.cos(3*angles)).flatten()
#
## Create the Triangulation; no triangles so Delaunay triangulation created.
#triang = mtri.Triangulation(x, y)
#
## Mask off unwanted triangles.
#xmid = x[triang.triangles].mean(axis=1)
#ymid = y[triang.triangles].mean(axis=1)
#mask = xmid**2 + ymid**2 < min_radius**2
#triang.set_mask(mask)
#
## Plot the surface.
#ax = fig.add_subplot(1, 2, 2, projection='3d')
#ax.plot_trisurf(triang, z, cmap=plt.cm.CMRmap)
#
plt.tight_layout()
plt.show()
