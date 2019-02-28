import cv2
import numpy as np
import matplotlib.pyplot as plt


def readMets(metfile):
    f=open(metricsfile,"r")
    lins = f.readlines()
    mets=[]

    for l in lins[1:]:
        l=l.strip().split(',')
        mets.append(l[1:])
    mets=np.asarray(mets, dtype=np.float)
    return mets


metricsfile = '/home/nader/scratch/mesh_test/new/comp_mets/crop/metrics.csv'
mets = readMets(metricsfile)

slope = mets[:,9]
Q3_slope = np.percentile(slope,75)

rug = mets[:,34]
Q3_rug = np.percentile(rug,75)

ht_range = mets[:,30]
Q3_htr = np.percentile(ht_range,75)

ht_std = mets[:,31]
Q3_hts = np.percentile(ht_std,75)

slp_quads = []
rug_quads = []
htr_quads = []
hts_quads = []
for m in mets:
    qd = m[2:4]
    if m[9]>Q3_slope:
        slp_quads.append(qd)

    if m[34]>Q3_rug:
        rug_quads.append(qd)

    if m[30]>Q3_htr:
        htr_quads.append(qd)

    if m[31]>Q3_hts:
        hts_quads.append(qd)
a=10

fig, ax = plt.subplots()
# ax.scatter(np.asarray(slp_quads)[:,1],np.asarray(slp_quads)[:,0],label='slope')
ax.scatter(np.asarray(rug_quads)[:,1],np.asarray(rug_quads)[:,0],label='rug')
# ax.scatter(np.asarray(htr_quads)[:,1],np.asarray(htr_quads)[:,0],label='htr')
ax.legend()
plt.show()

np.savetxt('/home/nader/scratch/mesh_test/new/comp_mets/crop/rugspots.csv',np.asarray(rug_quads))
