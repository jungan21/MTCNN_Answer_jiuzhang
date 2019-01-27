#!/bin/bash

cd prepare_data

# face detection data
python3 gen_12net_data.py

# landmark data
python3 gen_landmark_aug_12.py

# meage above 2 together
python3 gen_imglist_pnet.py

# convert to TF records data
python3 gen_PNet_tfrecords.py
