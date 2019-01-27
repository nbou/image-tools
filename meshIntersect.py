from plyfile import PlyData, PlyElement
import numpy as np
from sklearn.neighbors import KDTree
import time
import geoCropMesh
import matplotlib.pyplot as plt

# function to find the surface normal of a triangle
def triNorm(p1,p2,p3):
    N = np.cross(p2-p1,p3-p1)
    return N


# function to find the centroid of a triangle
def triCent(p1,p2,p3):
    x1,y1,z1 = p1
    x2,y2,z2 = p2
    x3, y3, z3 = p3

    C = np.array([(x1+x2+x3)/3,(y1+y2+y3)/3,(z1+z2+z3)/3])
    return C


# function to calculate the max distance between the centroid and vertices of a triangle
def triMaxDist(p1,p2,p3):
    C = triCent(p1,p2,p3)
    d = None
    for point in [p1,p2,p3]:
        dist = np.linalg.norm(C-point)
        if d == None or dist > d:
            d = dist
    return d


# function to convert a triangle (from a mesh) into a sphere. Returns the centre, radius and normal vector
def tri2sphere(p1,p2,p3):
    C = triCent(p1,p2,p3)
    nrm = triNorm(p1,p2,p3)
    rad = triMaxDist(p1,p2,p3)
    c1, c2, c3 = C
    n1, n2, n3 = nrm

    return np.array([c1,c2,c3, n1, n2, n3, rad])


# outputs an array with vertex 1 elements, normal and vertex indices of a triangle
def tri2triplus(inds, verts):
    i1,i2,i3 = inds
    p1 = verts[i1]
    p2 = verts[i2]
    p3 = verts[i3]
    # C = triCent(p1,p2,p3)
    nrm = triNorm(p1,p2,p3)
    # c1, c2, c3 = C
    n1, n2, n3 = nrm
    return np.array([p1[0],p1[1],p1[2], n1, n2, n3, int(i1),int(i2),int(i3)])

mesh = PlyData.read('/home/nader/scratch/mesh_test/final_crop.ply')
# mesh = PlyData.read('/home/nader/scratch/mesh_test/final.ply')
# read mesh into 3xn array
verts= np.transpose(np.array([mesh['vertex'].data['x'],mesh['vertex'].data['y'],mesh['vertex'].data['z']]))
# print(np.shape(verts))

tri_corners = np.array(mesh['face'].data['vertex_indices'])
mesh=None
# tree = KDTree(verts)

# # get spheres from triangles
# start = time.time()
# spheres = np.array([0,0,0,0,0,0,0])
# for corner in tri_corners:
#     p1 = verts[corner[0]]
#     p2 = verts[corner[1]]
#     p3 = verts[corner[2]]
#
#     sp = tri2sphere(p1,p2,p3)
#     spheres = np.vstack((spheres,np.array(sp)))
#
# spheres = spheres[1:][:]
# end = time.time()
# print('converted triangles into spheres in: ', end-start, ' seconds')

# get centre, normal from triangle indices
start = time.time()
tplus = np.array([0,0,0,0,0,0,0,0,0])
for corner in tri_corners:
    tplus = np.vstack((tplus,tri2triplus(corner,verts)))

tplus = tplus[1:][:]
end = time.time()
spheres = tplus
print('extracted triangle normals, centroids in: ', end-start, ' seconds')

# find square bounding the image points in the mesh
olon = 147.2306000000000097
olat = -43.6165000000000020
gtfpath = '/home/nader/scratch/PR_20100604_080817_570_LC16.tif'
# gtfpath = '/home/nader/scratch/PR_20100604_080818_584_LC16.tif'
ymin, ymax, xmin, xmax = geoCropMesh.meshCropPts(olon,olat,gtfpath)
# add buffer to bounds
buffer = np.float(0.5)
xmin = xmin - buffer*(xmax-xmin)
xmax = xmax + buffer*(xmax-xmin)
ymin = ymin - buffer*(ymax-ymin)
ymax = ymax + buffer*(ymax-ymin)


# xlim = verts[np.where(np.logical_and(np.greater_equal(xmax,verts[:,0]),np.less_equal(xmin, verts[:,0])))]
# ylim =  xlim[np.where(np.logical_and(np.greater_equal(ymax, xlim[:,2]),np.less_equal(ymin, xlim[:,2])))]

# crop spheres matrix using the corners from the chosen image
start=time.time()
xlim = spheres[np.where(np.logical_and(np.greater_equal(xmax,spheres[:,0]),np.less_equal(xmin, spheres[:,0])))]
lims =  xlim[np.where(np.logical_and(np.greater_equal(ymax, xlim[:,2]),np.less_equal(ymin, xlim[:,2])))]
end = time.time()
print('cropped sphere points to image corners in: ', end-start, ' seconds')

plt.scatter(spheres[:,0],spheres[:,2])
plt.scatter(lims[:,0],lims[:,2])
# plt.show()
print(np.shape(lims))
# print(lims)


# Find if/where a ray (generated from im2geo) intersects the spheres
ray_dir = np.array([0.43118462, 0.13044104, 1.00168327])
# ray_dir = np.array([ray_dir[2],ray_dir[0],ray_dir[1]])
ray_dir = ray_dir/np.linalg.norm(ray_dir)

ray_or = np.array([ -90.0693583,  -788.12009801,   69.17650223])
# ray_or = np.array([ray_or[0],ray_or[2],ray_or[1]])

for sp in spheres:
    x,y,z = [sp[0],sp[2],sp[1]]
    # print(x,y,z)
    pln_nrm = sp[3:6]
    pln_nrm = pln_nrm/np.linalg.norm(pln_nrm)
    pln_d = np.linalg.norm([x,y,z])
    p1 = np.array([x,y,z])
    p2 = verts[np.int(sp[7])]
    p3 = verts[np.int(sp[8])]
    pts = np.column_stack((p1,p2,p3))
    # calculate intersection point betweeen ray and triangle plane
    t = (-(np.dot(pln_nrm,ray_or))+ pln_d)/(np.dot(pln_nrm,pln_d))
    pln_ray_int = ray_or + ray_dir*t
    # check if point is inside triangle
    avs = np.matmul(np.linalg.inv(pts),pln_ray_int.reshape(3,1))
    print(avs.reshape(1,3)>0)
    plt.scatter(pln_ray_int[0],pln_ray_int[1])
    if (avs>0).all():
        print('hello')
        print(avs.reshape(1,3))
        # hello

plt.show()
