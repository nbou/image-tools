from plyfile import PlyData
import numpy as np
import time
import geoCropMesh
import matplotlib.pyplot as plt
import calParser
import os
from im2geo import im2geo

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

# read box positions from output of retinanet_inf_batch.py, filter detections with score less than filt
def readInf(inf_path, filt):
    f = open(inf_path, "r")
    lines = f.readlines()
    idx = 0

    out = []
    for line in lines[1:]:
        line = line.strip().split(',')
        if np.float(line[5]) >= filt:
            imnme = os.path.basename(line[0])[:-4]
            x1,y1,x2,y2 = np.array(line[1:5], dtype=np.float32)
            midpoint = np.array([(x2+x1)/2, (y2+y1)/2])

            out.append([idx,imnme,midpoint])
        idx+=1
    return out




# use osgcong x.ive y.ply to convert mesh into ply format first
mesh = PlyData.read('/home/nader/scratch/mesh_test/final.ply')
# mesh = PlyData.read('/home/nader/scratch/mesh_test/final.ply')
# read mesh into 3xn array
verts= np.transpose(np.array([mesh['vertex'].data['x'],mesh['vertex'].data['y'],mesh['vertex'].data['z']]))
# print(np.shape(verts))

tri_corners = np.array(mesh['face'].data['vertex_indices'])
print(np.shape(tri_corners))
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

inp = input('Load presaved mesh? (y/n) ')
if inp=='y':
    spheres = np.load('/home/nader/scratch/mesh_test/final.npy')
else:
    # get centre, normal from triangle indices
    start = time.time()
    tplus = np.array([0,0,0,0,0,0,0,0,0])
    for corner in tri_corners:
        tplus = np.vstack((tplus,tri2triplus(corner,verts)))

    tplus = tplus[1:][:]
    end = time.time()
    spheres = tplus
    print('extracted triangle normals, centroids in: ', end-start, ' seconds')

# find rectangle bounding the image points in the mesh
# olon = 147.2306000000000097
# olat = -43.6165000000000020
posepth = '/home/nader/scratch/stereo_pose_est.data'
olat,olon = calParser.getOrigin(posepth)

inf_path = '/home/nader/scratch/inf_boxes_huon_13.txt'

# read in inferences,
filt = 0.714  # minimum probability for detections
inflines = readInf(inf_path, filt)
# print(inflines[0])
plt.scatter(spheres[:, 0], spheres[:, 2])


intx=[]
for i in range(len(inflines)):
    print('reading detection {0} of {1}'.format(i,len(inflines)))
    geotif_dir = '/media/nader/ML_fish_data/lobsters/r20100604_061515_huon_13_deep_in2/renav20100605/i20100604_061515_gtif/'
    geotif_file = inflines[i][1] +'.tif'
    gtfpath = os.path.join(geotif_dir,geotif_file)#'/home/nader/scratch/PR_20100604_080817_570_LC16.tif'

    # gtfpath = '/home/nader/scratch/PR_20100604_080818_584_LC16.tif'
    ymin, ymax, xmin, xmax = geoCropMesh.meshCropPts(olon,olat,gtfpath)
    # add buffer to bounds
    buffer = np.float(0.75)
    xmin = xmin - buffer*(xmax-xmin)
    xmax = xmax + buffer*(xmax-xmin)
    ymin = ymin - buffer*(ymax-ymin)
    ymax = ymax + buffer*(ymax-ymin)


    # xlim = verts[np.where(np.logical_and(np.greater_equal(xmax,verts[:,0]),np.less_equal(xmin, verts[:,0])))]
    # ylim =  xlim[np.where(np.logical_and(np.greater_equal(ymax, xlim[:,2]),np.less_equal(ymin, xlim[:,2])))]

    # crop spheres matrix using the corners from the chosen image
    start=time.time()
    xlim = spheres[np.where(np.logical_and(np.greater_equal(xmax,spheres[:,0]),np.less_equal(xmin, spheres[:,0])))]
    lims = xlim[np.where(np.logical_and(np.greater_equal(ymax, xlim[:,2]),np.less_equal(ymin, xlim[:,2])))]
    end = time.time()
    # print('cropped sphere points to image corners in: ', end-start, ' seconds')

    # plt.scatter(spheres[:,0],spheres[:,2])
    # plt.scatter(lims[:,0],lims[:,2])
    # plt.show()
    # print(np.shape(lims))
    # print(lims)

    # R1,R2,R3
    # Find if/where a ray (generated from im2geo) intersects the spheres
    # ray_dir = np.array([0.05386494, 0.31301034,  0.99528809])#np.array([0.43118462, 0.13044104, 1.00168327])

    # inf_path = '/home/nader/scratch/inf_boxes_huon_13.txt'
    # inflines = readInf(inf_path)
    calfile = '/home/nader/scratch/ng2_scaled_baseline.calib'
    posefile = '/home/nader/scratch/stereo_pose_est.data'
    ray_dir, ray_or = im2geo(calfile,posefile, inflines[i])

    # ray_dir = np.array([-0.00169494, -0.00685659,  1.00000357])
    ray_dir = np.array([ray_dir[1],ray_dir[0],ray_dir[2]])
    # ray_dir = ray_dir/np.linalg.norm(ray_dir)

    # ray_or = np.array([185.29804681, -548.70514385,   53.60263709])#np.array([ -90.0693583,  -788.12009801,   69.17650223])
    ray_or = np.array([ray_or[1],ray_or[0],ray_or[2]])

    # # (R3,R2),R1
    # # Find if/where a ray (generated from im2geo) intersects the spheres
    # ray_dir = np.array([0.05840763, 0.31392563, 1.00327564])#np.array([0.43118462, 0.13044104, 1.00168327])
    # ray_dir = np.array([ray_dir[1],ray_dir[0],ray_dir[2]])
    # ray_dir = ray_dir/np.linalg.norm(ray_dir)
    #
    # ray_or = np.array([185.29804681, -548.70514385,   53.60263709])#np.array([ -90.0693583,  -788.12009801,   69.17650223])
    # ray_or = np.array([ray_or[1],ray_or[0],ray_or[2]])

    # go through each triangle in the mesh and check if the ray intersects it
    num_ints = 0
    # intx = []
    for sp in lims:
        x,y,z = [sp[0],sp[2],sp[1]]
        # print(x,y,z)
        pln_nrm = sp[3:6]   # extract normal for triangle
        pln_nrm = np.array([pln_nrm[0],pln_nrm[2],pln_nrm[1]])  # shuffle the xyz coords to match the ray
        # print(sp, pln_nrm)
        pln_nrm = pln_nrm/np.linalg.norm(pln_nrm)   # normalise triangle normal
        pln_d = np.linalg.norm([x,y,z])   # d in plane equation (distance from point to (0,0,0))
        # specify triangle vertices p1,p2,p3
        p1 = np.array([x,y,z])
        p2 = verts[np.int(sp[7])]
        p2 = np.array([p2[0],p2[2],p2[1]])
        p3 = verts[np.int(sp[8])]
        p3 = np.array([p3[0], p3[2], p3[1]])

        # calculate intersection point between ray (origin + t*direction) and triangle plane
        t_denom = (np.dot(ray_dir,pln_nrm))
        pln_ray_dist = p1 - ray_or
        t = np.divide(np.dot(pln_ray_dist,pln_nrm),t_denom)
        pln_ray_int = ray_or + ray_dir*t

        # check if point is inside triangle (using barycentric method http://blackpawn.com/texts/pointinpoly/)
        v0 = p3 - p1
        v1 = p2 - p1
        v2 = pln_ray_int - p1
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)
        invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom
        if (u >=0) and (v>=0) and (u+v<1):
            # print(p1)
            # print(p2)
            # print(p3)
            # print(pln_ray_int)
            # print('hello')
            #
            plt.scatter(pln_ray_int[0],pln_ray_int[1])
            num_ints+=1
            intx.append(pln_ray_int)

    #
    #
plt.show()
# print(num_ints)
np.save('/home/nader/scratch/lobsterpts_3d', intx)