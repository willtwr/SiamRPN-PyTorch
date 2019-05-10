#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
from glob import glob
from multiprocessing.pool import ThreadPool

import numpy as np
import cv2
from cv2 import imread, imwrite
from PIL import Image
from shutil import copyfile

CURRENT_DIR = osp.dirname(__file__)
ROOT_DIR = osp.join(CURRENT_DIR, '..')
sys.path.append(ROOT_DIR)

from utils.infer_utils import get_crops, Rectangle, convert_bbox_format
from utils.misc_utils import mkdir_p


def process_split(root_dir, save_dir, split):
  data_dir = osp.join(root_dir, split)
  video_names = os.listdir(data_dir)
  video_names = [vn for vn in video_names if '.txt' not in vn]

  for idx, video in enumerate(video_names):
    print('{split} ({idx}/{total}): Processing {video}...'.format(split=split,
                                                                  idx=idx, total=len(video_names),
                                                                  video=video))
    video_path = osp.join(data_dir, video)
    jpg_files = glob(osp.join(video_path, '*.jpg'))
    
    with open(osp.join(video_path, 'groundtruth.txt')) as f:
      ann_content = f.readlines()
      
    track_save_dir = osp.join(save_dir, split, video)
    mkdir_p(track_save_dir)
    fw = open(osp.join(track_save_dir, 'groundtruth.txt'), 'w')
    
    copyfile(osp.join(video_path, 'absence.label'), osp.join(track_save_dir, 'absence.label'))
    copyfile(osp.join(video_path, 'cover.label'), osp.join(track_save_dir, 'cover.label'))
    copyfile(osp.join(video_path, 'cut_by_image.label'), osp.join(track_save_dir, 'cut_by_image.label'))
    copyfile(osp.join(video_path, 'meta_info.ini'), osp.join(track_save_dir, 'meta_info.ini'))
    
    for i, jpg in enumerate(jpg_files):
      # Read image
      img_file = jpg.split('/')[-1]
      img = None
      
      # Get all object bounding boxes
      jpgidx = img_file.split('.')[0]
      jpgidx = int(jpgidx) - 1
      ann = ann_content[jpgidx]
      ann = ann.strip()
      bbox = ann.split(',')
      bbox = [int(float(bb)) for bb in bbox]  # [xmin, ymin, w, h]
      
      ## bbox ####
      annk = ann_content[i]
      annk = annk.strip()
      bboxk = annk.split(',')
      bboxk = [int(float(bb)) for bb in bboxk]  # [xmin, ymin, w, h]      
      
      w = bboxk[2]
      h = bboxk[3]
      context_amount = 0.5
      size_z=127
      size_x=271
      
      wc_z = w + context_amount * (w + h)
      hc_z = h + context_amount * (w + h)
      s_z = np.sqrt(wc_z * hc_z)
      scale_z = size_z / s_z
      d_search = (size_x - size_z) / 2
      pad = d_search / scale_z
      s_x = s_z + 2 * pad
      
      wn = int(w * size_x / s_x)
      hn = int(h * size_x / s_x)
      
      #if wn < 1 or hn < 1:
        #if wn == 0:
          #wn = 1
        #if hn == 0:
          #hn = 1
        #ratio = wn / hn
        #if ratio > 1.:
          #newbb = [int(135-wn/2), int(135-hn/2), 85, int(85. / ratio)]
        #else:
          #newbb = [int(135-wn/2), int(135-hn/2), int(85. * ratio), 85]
      #else:
        #newbb = [int(135-wn/2), int(135-hn/2), wn, hn]
        
      if wn < 1:
        wn = 1
      if hn < 1:
        hn = 1      
      newbb = [int(135-wn/2), int(135-hn/2), wn, hn]
        
      fw.write(','.join(str(e)+'.0000' for e in newbb)+'\n')
      ## bbox ####
      
      savename = osp.join(track_save_dir, '{}.jpg'.format(img_file))
      
      if osp.isfile(savename): 
        try:
          im = Image.open(savename)
          continue  # skip existing images
        except IOError:
          os.remove(savename)
          
      if img is None:
        img = imread(jpg)
        
      # Get crop
      target_box = convert_bbox_format(Rectangle(*bbox), 'center-based')
      crop, _ = get_crops(img, target_box,
                          size_z=127, size_x=271,
                          context_amount=0.5)
      imwrite(savename, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    
    fw.close()


if __name__ == '__main__':
  vid_dir = 'data/GOT-10k'

  # Or, you could save the actual curated data to a disk with sufficient space
  # then create a soft link in `data/ILSVRC2015-VID-Curation`
  save_dir = 'data/got10k-Curation'

  pool = ThreadPool(processes=2)

  one_work = lambda a: process_split(vid_dir, save_dir, a)

  results = []
  results.append(pool.apply_async(one_work, ['val']))
  results.append(pool.apply_async(one_work, ['train']))
  ans = [res.get() for res in results]
