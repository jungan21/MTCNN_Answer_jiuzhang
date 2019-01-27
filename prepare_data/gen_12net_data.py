#coding:utf-8
import sys
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU
anno_file = "wider_face_train.txt" # train data 的 label
im_dir = "WIDER_train/images" # train data 实际的图像存储路径
pos_save_dir = "12/positive" # 1
part_save_dir = "12/part" # -1 i.e. 框 和 ground truth IOU 在 0.45 0.6 之间
neg_save_dir = '12/negative' # 0 
save_dir = "./12"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(pos_save_dir):
    os.mkdir(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.mkdir(part_save_dir)
if not os.path.exists(neg_save_dir):
    os.mkdir(neg_save_dir)

f1 = open(os.path.join(save_dir, 'pos_12.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_12.txt'), 'w')
f3 = open(os.path.join(save_dir, 'part_12.txt'), 'w')
with open(anno_file, 'r') as f: # wider_face_train.txt label file
    annotations = f.readlines()
num = len(annotations)
print("%d pics in total" % num)
p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0

"""
 annotation(i.e. label file) format: http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/ 
 each line in label file (i.e.wider_face_train.txt): 0--Parade/0_Parade_marchingband_1_849 448.51 329.63 570.09 478.23 448.51 329.63 570.09 478.23
"""
for annotation in annotations: # iterate through all label lines
    annotation = annotation.strip().split(' ')
    im_path = annotation[0] #image path
    #boxed change to float type
    bbox = list(map(float, annotation[1:])) # 用' ' split之后，从第一个元素开始往后的数字都是表示ground truth bounding box的位置的数字
    #gt
    # 用' ' split之后, annotation[1:]，每4个数字为一组， 表示一个bounding box的位置
    # sample:448.51 329.63 570.09 478.23 448.51 329.63 570.09 478  每四个数字reshape成一行，表示一个gt bounding box 的位置
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4) 
    #load image
    img = cv2.imread(os.path.join(im_dir, im_path + '.jpg')) # im_dir = "WIDER_train/images", im_path="0--Parade/0_Parade_marchingband_1_799"
    
    # just for tracking progress, 每处理100个图片，输出一个log
    idx += 1
    if idx % 100 == 0:
        print("%d images done" % idx)
        
    height, width, channel = img.shape

    neg_num = 0
    #1---->50
    while neg_num < 50:
        #neg_num's size [40,min(width, height) / 2],min_size:40 
        size = npr.randint(12, min(width, height) / 2)
        #top_left
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        #random crop
        crop_box = np.array([nx, ny, nx + size, ny + size])
        #cal iou
        Iou = IoU(crop_box, boxes)
        
        cropped_im = img[ny : ny + size, nx : nx + size, :]
        resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
            f2.write("12/negative/%s.jpg"%n_idx + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1
    #as for 正 part样本
    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        #gt's width
        w = x2 - x1 + 1
        #gt's height
        h = y2 - y1 + 1

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue
        for i in range(5):
            size = npr.randint(12, min(width, height) / 2)
            # delta_x and delta_y are offsets of (x1, y1)
            delta_x = npr.randint(max(-size, -x1), w)
            delta_y = npr.randint(max(-size, -y1), h)
            nx1 = int(max(0, x1 + delta_x))
            ny1 = int(max(0, y1 + delta_y))
            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            Iou = IoU(crop_box, boxes)
    
            cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
    
            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write("12/negative/%s.jpg" % n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1        
	# generate positive examples and part faces
        for i in range(20):
            # pos and part face size [minsize*0.8,maxsize*1.25]
            size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # delta here is the offset of box center
            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)
            #show this way: nx1 = max(x1+w/2-size/2+delta_x)
            nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
            #show this way: ny1 = max(y1+h/2-size/2+delta_y)
            ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue 
            crop_box = np.array([nx1, ny1, nx2, ny2])
            #yu gt de offset
            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)
            #crop
            cropped_im = img[ny1 : ny2, nx1 : nx2, :]
            #resize
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            if IoU(crop_box, box_) >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                f1.write("12/positive/%s.jpg"%p_idx + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif IoU(crop_box, box_) >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                f3.write("12/part/%s.jpg"%d_idx + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
        box_idx += 1
print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))
f1.close()
f2.close()
f3.close()
