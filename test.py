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
    filename = save_folder + 'result.json'
    with open(filename, mode='a') as f:
      f.write('[')
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                            img.shape[1], img.shape[0]])

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
            
    with open(filename, mode='rb+') as f:
      f.seek(-1, os.SEEK_END)
      f.truncate()
    with open(filename, mode='a') as f:
      f.write(']')
def test_voc():
# load net
    num_classes = 9 # change
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = COCODetection(args.coco_root, 'val35k', None, COCOAnnotationTransform)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
    BaseTransform(net.size, (104, 117, 123)),
    thresh=args.visual_threshold)

if __name__ == "__main__":
  test_voc()
