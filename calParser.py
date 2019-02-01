import numpy as np

def parseCal(filepth):

    # filepth = '/home/nader/scratch/ng2_scaled_baseline.calib'

    f = open(filepth)
    lins = f.readlines()

    c1 = lins[2].split(' ')
    c1_mat = np.array([c1[2:5], c1[5:8], c1[8:11]],dtype='float64')
    c1_dist = np.array(c1[11:15],dtype='float64' )
    return c1_mat, c1_dist
