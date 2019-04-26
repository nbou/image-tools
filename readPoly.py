from osgeo import ogr
import matplotlib.pyplot as plt

shpPth = '/home/nader/scratch/palau/mosaic/pgons.shp'

file = ogr.Open(shpPth)
shape = file.GetLayer(0)
fCount = shape.GetFeatureCount() # get the number of polygons

# extract (and plot) the vertices of each polygon
for f in range(fCount):
    feature = shape.GetFeature(f)
    # first = feature.ExportToJson()
    # print(first) # (GeoJSON format)
    geom = feature.GetGeometryRef()
    ring = geom.GetGeometryRef(0)
    points = ring.GetPointCount()


    lon = []
    lat = []
    for p in range(points):
        lo,la,z = ring.GetPoint(p)
        lon.append(lo)
        lat.append(la)

    print(len(lon))
    print(len(lat))
    plt.plot(lon,lat)
plt.show()