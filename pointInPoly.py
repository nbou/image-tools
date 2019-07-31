import numpy as np
from readPolyFirst import readPolyFirst
from plyfile import PlyData
import matplotlib.pyplot as plt

import time

# function to find the smallest box containing a polygon from its vertices
def polyBox(x,y):
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    box = xmin,xmax,ymin,ymax
    return box

# read vertices of polygon
lon,lat = readPolyFirst()

# calculate equations for each of the line segments in the polygon
m=[] # gradients
b=[] # y intercepts
lon_lims = [] # x range for each line segment
lat_lims = []
for i in range(len(lon)-1):
    print(i)
    grad = (lat[i+1]-lat[i])/(lon[i+1] - lon[i])
    y_int = lat[i+1]- grad*lon[i+1]
    m.append(grad)
    b.append(y_int)
    lons = [lon[i],lon[i+1]]
    lats = [lat[i],lat[i+1]]
    lon_lims.append([np.min(lons), np.max(lons)])
    lat_lims.append([np.min(lats), np.max(lats)])

a=1
# box = polyBox(lon,lat)
# print(box)
# # read in mesh points
meshPth = '/home/nader/scratch/palau/pats_02/model_test.ply'
# # meshPth = '/home/nader/scratch/palau/mosaic/total.ply'
# tic = time.time()
mesh = PlyData.read(meshPth)
verts= np.transpose(np.array([mesh['vertex'].data['x'],mesh['vertex'].data['y'],mesh['vertex'].data['z']]))
print(np.shape(verts))
plt.scatter(mesh['vertex'].data['x'],mesh['vertex'].data['y'])
plt.plot(lon,lat,'r')
# plt.axis([134.5271133605008, 134.5271695118497, 7.280636960912066, 7.280664111938539])
#
plt.show()

print(verts[0])

