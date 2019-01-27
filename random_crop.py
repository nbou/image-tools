import cv2
import numpy as np
from random import randint

def randcrop(im, boxsize):
    buff = boxsize/2 + 1
    mid_x = randint(buff, np.shape(im)[0] - buff)
    mid_y = randint(buff, np.shape(im)[1] - buff)

    crp = im[mid_x - boxsize/2 : mid_x + boxsize/2,
          mid_y - boxsize/2 : mid_y + boxsize/2, :]
    return crp, mid_x, mid_y

mospath = 'test_data/test.tif'
mosaic = cv2.imread(mospath)


for i in range(30):
    crp, mx, my = randcrop(mosaic, 500)
    if np.sum(crp)/(np.prod(np.shape(crp))*255) <= 0.8:
        # cv2.imshow('im', crp)
        # cv2.waitKey(0)
        fname = str(int(mx)) + '_' + str(int(my)) + '.jpg'
        cv2.imwrite('test_data/output/' + fname, crp)






