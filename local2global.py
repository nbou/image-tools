# script to convert a tab delimited csv of coordinates from an auv mission into lat/longs. The origin lat/long can be
# found in stereo_pose_est.data

import pyproj
import numpy as np
import pymap3d

def reProj(x,y, olon, olat):
# geoloc = pymap3d.ned2geodetic(image['pose']['position'][0],
#                                       image['pose']['position'][1],
#                                       image['pose']['position'][2],
#                                       datum['latitude'],
#                                       datum['longitude'],
#                                       datum['altitude'], ell=None, deg=True)
    geoloc = pymap3d.ned2geodetic(x,
                                  y,
                                  0,
                                  olat,
                                  olon,
                                  0, ell=None, deg=True)
    return geoloc[0],geoloc[1]

coordPth = '/home/nader/scratch/St_helens_is_2009_night_deep/mesh/comp_2_0.5/quad_cents.csv'

f = open(coordPth, "r")


olon = 148.3412500000000023
olat = -41.3438170000000014
for line in f.readlines():
    l = line.strip().split(',')
    x,y = reProj(float(l[0]),float(l[1]), olon, olat)
    print(x,y)