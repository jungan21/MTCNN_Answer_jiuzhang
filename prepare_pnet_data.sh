#!/bin/bash

cd prepare_data

# 生成 face detection training data: 人脸数据: 用来生成  正，负， part 样本 的training data
python3 gen_12net_data.py

# 生成 landmark training data
python3 gen_landmark_aug_12.py

# meage above 2 together
python3 gen_imglist_pnet.py

# convert to TF records data
python3 gen_PNet_tfrecords.py
