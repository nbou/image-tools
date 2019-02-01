import numpy as np

def parseCal(filepth):

    # filepth = '/home/nader/scratch/ng2_scaled_baseline.calib'

    f = open(filepth)
    lins = f.readlines()

    c1 = lins[2].split(' ')
    c1_mat = np.array([c1[2:5], c1[5:8], c1[8:11]], dtype='float64')
    c1_dist = np.array(c1[11:15], dtype='float64')
    return c1_mat, c1_dist


def getOrigin(filepth):
    f = open(filepth)
    lins = f.readlines()
    lat = lins[55].strip().split(' ')[2]
    lon = lins[56].strip().split(' ')[1]
    return(lat,lon)


def parsePose(stereo_pose_file, im):
    # filepth ='/home/nader/scratch/stereo_pose_est.data'
    #im = 'PR_20100604_062303_835_LC16'
    f = open(stereo_pose_file)
    lins = f.read().splitlines()
    lins = lins[57:]


    for l in lins:
        # print(l)
        if l.split('\t')[10][:27] == im:
            pose = np.array(l.split('\t')[:10],dtype='float64')
            break

    return pose


pose = parsePose('/home/nader/scratch/stereo_pose_est.data', 'PR_20100604_062303_835_LC16')
print(pose)
