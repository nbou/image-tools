import numpy as np
import os

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
            score = line[-1]
            out.append([idx,imnme,midpoint,score])
        idx+=1
    return out


inf_path = '/home/nader/scratch/inf_boxes_huon_13.txt'

# read in inferences,
filt = 0  # minimum probability for detections
inflines = readInf(inf_path, filt)
filt = 0.8224
# add 1 to point index to get correct line in inf_boxes

# ind_path = '/home/nader/scratch/lobsterpt_indices'
# f = open(ind_path,"r")
# indlines = f.readlines()
#
# loc_path = '/home/nader/scratch/pt_locs.txt'
# f = open(loc_path, "r")
# loclines = f.readlines()

# inds = []
# out = []
# for l in indlines:
#     pt_idx = int(l.strip())
#     im_idx = pt_idx + 1
#     pt = [float(i) for i in loclines[pt_idx].strip().split(' ')]
#     out.append([[pt_idx, inflines[im_idx],1], [float(i) for i in loclines[pt_idx].strip().split(' ')]])

inds = np.load('/home/nader/scratch/lobsterpt_indices.npy')
pts = np.load('/home/nader/scratch/lobsterpts_3d.npy')
# inds = np.array(inds_pts[:,0],dtype=np.int)
# pts = np.array(inds_pts[:,2:],dtype=np.float)

xmin=-520
xmax = -440
ymin = 60
ymax = 140

outfile = '/home/nader/scratch/cropped_inf_boxes.txt'
f=open(outfile,"w+")
inds_cropped=[]
for i in range(len(inds)):
    x=pts[i][0]
    y=pts[i][1]
    if xmin <= x <= xmax and ymin <= y <=ymax and np.float(inflines[i][-1]) >= filt:
        inds_cropped.append(inflines[inds[i]])
        lne = str(inflines[i][0]) + ',' + str(inflines[i][1]) + ',' + str(inflines[i][2][0]) + ',' + str(inflines[i][2][1]) + '\n'
        f.write(lne)

