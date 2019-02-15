from plyfile import PlyData
import numpy as np
import time


# function to find the surface normal of a triangle
def triNorm(p1,p2,p3):
    N = np.cross(p2-p1,p3-p1)
    return N


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

def ply2norms(meshpth, outpth):
    # use osgcong x.ive y.ply to convert mesh into ply format first
    # mesh = PlyData.read('/home/nader/scratch/mesh_test/final_crop.ply')
    # mesh = PlyData.read('/home/nader/scratch/mesh_test/final.ply')
    mesh=PlyData.read(meshpth)

    # read mesh into 3xn array
    verts= np.transpose(np.array([mesh['vertex'].data['x'],mesh['vertex'].data['y'],mesh['vertex'].data['z']]))
    # print(np.shape(verts))

    tri_corners = np.array(mesh['face'].data['vertex_indices'])
    # print(np.shape(tri_corners))
    mesh=None

    start = time.time()
    tplus = np.array([0,0,0,0,0,0,0,0,0])
    for corner in tri_corners:
        tplus = np.vstack((tplus,tri2triplus(corner,verts)))

    tplus = tplus[1:][:]
    end = time.time()
    spheres = tplus
    print('extracted triangle normals, centroids in: ', end-start, ' seconds')
    q = input('save output array? (y/n) ')
    if q=='y':
        np.save(outpth,spheres)
    return 0


# meshpth = '/home/nader/scratch/mesh_test/final_crop.ply'
# outpth = '/home/nader/scratch/mesh_test/final_crop'

meshpth = '/home/nader/scratch/mesh_test/final.ply'
outpth = '/home/nader/scratch/mesh_test/final'
ply2norms(meshpth,outpth)