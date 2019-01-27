import os

# function to label file with boxes of given size centred around provided image points
def csvwrite(outputpath, imgpath, boxsize, objects):
    # check if labels file exists. If it does, new labels are added an new rows, if not, the
    # file is created
    if os.path.exists(outputpath):
        appwri = "a"
    else:
        appwri = "w"

    f = open(outputpath, appwri)
    for obj in objects:
        box = [obj[0]- boxsize/2, obj[1] - boxsize/2, obj[0] + boxsize/2, obj[1] + boxsize/2]
        f.write(imgpath + ',' + str(box[0]) + ',' + str(box[1]) + ',' + str(box[2]) + ',' + str(box[3]) + ',head\n')
    f.close()

# given a file, read all the lines getting rid of the top line (i.e. the header)
def readAnno(Pth):
    lines = [line.split('\t') for line in open(Pth)]
    lines = lines[1:]
    return lines



