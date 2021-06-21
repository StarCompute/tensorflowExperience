import os

from tensorflow.core.example.feature_pb2 import Feature, Features
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import glob
import io
import xml.etree.ElementTree as ET
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
# from collections import namedtuple,OrderedDict


def getClass(cname):
    if cname=='mouse':
        return 1
    elif cname=='frog':
        return 2
    elif cname=='contrl':
        return 3
    else:
        return 99

xmlpath='c:/opencv/test/pos2/'
output_filename="C:/opencv/test/data2/train.tfrecord"
# output_pbxt="C:/opencv/test/data2/train.pbtxt"

writer=tf.io.TFRecordWriter(output_filename)
# tf.io.TFRecordWriter()
pbtx=''
id=1
for xmlfile in glob.glob(xmlpath+"*.xml"):
    # print(xmlfile)
    tree=ET.parse(xmlfile)
    root=tree.getroot()
    filename=root.find('filename').text
    size=root.find('size')
    width=int(size.find('width').text)
    height=int(size.find('height').text)
    rect=root.find('object/bndbox')
    xmin=int(rect.find('xmin').text)
    ymin=int(rect.find('ymin').text)
    xmax=int(rect.find('xmax').text)
    ymax=int(rect.find('ymax').text)
    image_format=b'jpg'
    classname=root.find('object/name').text
    classlabel=getClass(classname)
    filepath=root.find('path').text
    with tf.compat.v1.gfile.GFile(filepath, 'rb') as fid:
        encoded_jpg = fid.read()



    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes_label = []
    xmins.append(xmin/width)
    xmaxs.append(xmax/width)
    ymins.append(ymin/height)
    ymaxs.append(ymax/height)
    classes_text.append(classname.encode('utf8'))
    classes_label.append(classlabel)


    tf_example=tf.train.Example(
        features=tf.train.Features(feature={
            'image/height':dataset_util.int64_feature(height),
            'image/width':dataset_util.int64_feature(width),
            'image/filename':dataset_util.bytes_feature(filename.encode('utf8')),
            'image/source_id':dataset_util.bytes_feature(filename.encode('utf8')),
            'image/encoded':dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes_label),
        })
    )
    pbtx+='item{\n'
    # pbtx+='    id:'+str(id)+'\n'
    pbtx+='    id:'+str(classes_label)+'\n'
    
    pbtx+='    name:\''+classname+'\'\n'
    pbtx+='}'+'\r\n'
    id=id+1
    writer.write(tf_example.SerializeToString())



    print (filepath,filename,width,height,classname,classlabel,xmins,ymins,xmaxs,ymaxs,image_format,xmax/width)

writer.close()
