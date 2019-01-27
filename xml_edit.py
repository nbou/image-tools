from xml.dom.minidom import parse
import os
import sys

addr = str(sys.argv[1])

for i in range(len([name for name in os.listdir('.') if os.path.isfile(name)])-1):

    # create a backup of original file
    new_file_name = ('%d.xml'%(i+1))
    #old_file_name = new_file_name[:1]+'~'+new_file_name[1:]
    #os.rename(new_file_name, old_file_name)

    # change text value of element
    doc = parse(new_file_name)
    node = doc.getElementsByTagName('path')
    path_name = ('%s/%d.jpg'%(addr,i+1))
    node[0].firstChild.nodeValue = path_name

    node = doc.getElementsByTagName('filename')
    file_name = ('%d.jpg' % (i+1))
    node[0].firstChild.nodeValue = file_name

    node = doc.getElementsByTagName('folder')
    node[0].firstChild.nodeValue = 'VOC2012'

    # persist changes to new file
    xml_file = open(new_file_name, "w")
    doc.writexml(xml_file, encoding="utf-8")
    xml_file.close()

