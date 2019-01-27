from osgeo import gdal, osr
import numpy as np
import pyproj

def GetExtent(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
            # print(x,y)
        yarr.reverse()
    return ext

def ReprojectCoords(coords,src_srs,tgt_srs):
    ''' Reproject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords

# origin long and lat (can be found in renav*/vehicle_pose_est.data file
olon = 147.2306000000000097
olat = -43.6165000000000020
# define projection p between lat/long and local coords
projStr = '+proj=tmerc +lon_0={} +lat_0={} +units=m'.format(olon, olat)
p = pyproj.Proj(projStr)

# read in geotif (found renav*/i*gtif
gtfpath = '/home/nader/scratch/PR_20100604_080817_570_LC16.tif'
ds = gdal.Open(gtfpath)
gt=ds.GetGeoTransform()
cols = ds.RasterXSize
rows = ds.RasterYSize
ext=GetExtent(gt,cols,rows)

src_srs=osr.SpatialReference()
src_srs.ImportFromWkt(ds.GetProjection())
#tgt_srs=osr.SpatialReference()
#tgt_srs.ImportFromEPSG(4326)
tgt_srs = src_srs.CloneGeogCS()

# lat longs of image corners
geo_ext=ReprojectCoords(ext,src_srs,tgt_srs)

# reproject geo_ext into local coordinate frame
# y, x = p(longitude, latitude)

corners = np.array([0,0])
for el in geo_ext:
    lon,lat = el
    y,x = p(lon,lat)
    corners = np.vstack((corners,np.array([x,y])))
    # print(y,x)
corners = corners[1:][:]
xmin, ymin = np.min(corners,axis=0)
xmax,ymax  = np.max(corners,axis=0)
print(xmin,xmax,ymin,ymax)
