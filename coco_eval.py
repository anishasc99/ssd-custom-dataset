#%matplotlib inline
from __future__ import print_function
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os, sys, zipfile
import urllib.request
import shutil
import numpy as np
import skimage.io as io
import pylab
import json
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

# Record package versions for reproducibility
print("os: %s" % os.name)
print("sys: %s" % sys.version)
print("numpy: %s, %s" % (np.__version__, np.__file__))

annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print('Running demo for *%s* results.'%(annType))

# Setup data paths
dataDir = os.getcwd()+'/data'
dataType = 'val2014'
annDir = '{}/annotations'.format(dataDir)
annZipFile = '{}/annotations_test{}.zip'.format(dataDir, dataType)
#annFile = 'eval/result.json'
annFile = 'data/annotations/instances_val35k.json'
annURL = 'http://images.cocodataset.org/annotations/annotations_test{}.zip'.format(dataType)
print (annDir)
print (annFile)
print (annZipFile)
#print (annURL)

# Download data if not available locally
if not os.path.exists(annDir):
    os.makedirs(annDir)
'''if not os.path.exists(annFile):
    if not os.path.exists(annZipFile):
        print ("Downloading zipped annotations to " + annZipFile + " ...")
        with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
            shutil.copyfileobj(resp, out)
        print ("... done downloading.")
    print ("Unzipping " + annZipFile)
    with zipfile.ZipFile(annZipFile,"r") as zip_ref:
        zip_ref.extractall(dataDir)
    print ("... done unzipping")'''
print ("Will use annotations in " + annFile)

cocoGt=COCO(annFile)

#initialize COCO detections api
resFile='%s/results/%s_%s_fake%s100_results.json'
resFile = resFile%(dataDir, prefix, dataType, annType)
resFile = 'eval/result.json'
cocoDt=cocoGt.loadRes(resFile)

imgIds=sorted(cocoGt.getImgIds())
#print(imgIds)
imgIds=imgIds[0:35]
imgId = imgIds[np.random.randint(len(imgIds)%100)]

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
