from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import COCO_ROOT, COCO_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from data import COCO_CLASSES, COCOAnnotationTransform, COCODetection
import torch.utils.data as data
from ssd import build_ssd
import json
import cv2
import pytesseract
from pytesseract import Output

'''COCO_change_category = ['0','1','2','3','4','5','6','7','8','9','10','11','13','14','15','16','17','18','19','20',
'21','22','23','24','25','26','27','28','31','32','33','34','35','36','37','38','39','40',
'41','42','43','44','46','47','48','49','50','51','52','53','54','55','56','57','58','59',
'60','61','62','63','64','65','67','70','72','73','74','75','76','77','78','79','80','81',
'82','84','85','86','87','88','89','90']
'''
COCO_change_category = ['0','1','2','3','4','5','6','7','8']
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_COCO_395000.pth',
type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.25, type=float,
help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
help='Use cuda to train model')
parser.add_argument('--coco_root', default=COCO_ROOT, help='Location of VOC root directory')
parser.add_argument('--image', default="11003-Ravi-Singh-Baroda-Gujarat-1_jpg.rf.bd00f49465c1537175003f7c866c64ac.jpg", help='Location of test image file')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
  torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
  torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
  os.mkdir(args.save_folder)

def test_net(save_folder, net, cuda, testset, transform, thresh):
# dump predictions and assoc. ground truth to text file for now
    filename = save_folder + 'result_test.txt'
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        os.environ['DISPLAY'] = ':0'
        PATH = COCO_ROOT+"/images/test35k/"+args.image
        newfile = COCO_ROOT+"/images/test35k/"+"resized_img.jpg"
        test_img = cv2.imread(PATH)  #original untampered image

        #test_img = cv2.resize(test_img, None, fx=4, fy=6)
        im = Image.open(PATH)
        im = im.resize((416,416))  #resize image to size of training data
        im.save(newfile, dpi=(300,300))
        img = cv2.imread(newfile) #read resized image

        test_img1 = cv2.imread(newfile) #keep 2 separate files: one for marking rectangles and another for ocr with no rectangle markings
        #test_img1 = cv2.resize(test_img1, (416,416))
        test_img1 = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([test_img.shape[1], test_img.shape[0],
                            test_img.shape[1], test_img.shape[0]])
        
        # ii -> category id
        for ii in range(detections.size(1)):
            j = 0
            while detections[0, ii, j, 0] >= thresh:

                score = detections[0, ii, j, 0].cpu().data.numpy()
                pt = (detections[0, ii, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])

                # standard format of coco ->
                # [{"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236},{...},...]
                with open(filename, mode='a') as f:
                    f.write(
                        '{"image_id":' + str(testset.pull_anno(i)[0]['image_id']) +
                        ',"category_id":' + str(COCO_change_category[ii]) +
                        ',"bbox":[' + ','.join(str(c) for c in coords) + ']'
                        ',"score":')
                    f.write('%.2f' %(score))
                    f.write('},')
                    # you need to delete the last ',' of the last image output of test image
                j += 1

                #cv2.rectangle(test_img, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]),(255,0,0), 2)
                x1 = int(coords[0])
                y1 = int(coords[1])
                x2 = int(coords[2])
                y2 = int(coords[3])
                crop_img = test_img1[y1:y2, x1:x2]

                #crop_img = cv2.resize(crop_img, None, fx=5, fy=5)
                cv2.imwrite("cropped.jpg",crop_img)
                custom_oem_psm_config = r'--oem 1'
                d = pytesseract.image_to_data(crop_img, output_type=Output.DICT,config = custom_oem_psm_config)
                print(d['text'])

                cv2.rectangle(test_img, (coords[0], coords[1]), (coords[2], coords[3]),(255,0,0), 2)
                cv2.putText(test_img, str(COCO_change_category[ii]), (int(coords[0]),int( coords[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (36,255,12), 1)
                
                
        cv2.imwrite("test_img.jpg",test_img)

def test_voc():
# load net
    num_classes = 9 # change
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    # add dummy annotations because COCODetection() by default requires for json file with annotations
    annotation_path = COCO_ROOT+'/annotations/instances_test35k.json'
    with open(annotation_path) as f:
      annotation = json.load(f)
      data = annotation['images'][0]
      data["file_name"] = args.image
      annotation['images'][0] = data
      print(annotation)
      with open(annotation_path, 'w') as fout:
        json_dumps_str = json.dumps(annotation, indent=4)
        print(json_dumps_str, file=fout)
    testset = COCODetection(args.coco_root, 'test35k', None, COCOAnnotationTransform)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
    BaseTransform(net.size, (104, 117, 123)),
    thresh=args.visual_threshold)

if __name__ == "__main__":
  test_voc()
