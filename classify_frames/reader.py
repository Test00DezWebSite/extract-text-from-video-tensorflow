import os, sys
import cv2
import pickle
import numpy as np
from scipy import misc
import struct
import six
from six.moves import urllib, range
import copy
import logging

from tensorpack import *
from cfgs.config import cfg

#from morph import warp

def get_center_frame_list(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [ele.strip() for ele in content]
    return content

def read_data(content):
    center_frame_path, label = content.split(' ')
    label = int(label)
    dir_path = '/'.join(center_frame_path.split('/')[:-1])
    center_frame_idx = int(center_frame_path.split('/')[-1].split('.')[0])
    num_frames = len(os.listdir(dir_path)) - 1
    margin = len(cfg.frame_extract_pattern) // 2
    if center_frame_idx - margin < 0 or center_frame_idx + margin >= num_frames:
        return

    frames_path = [os.path.join(dir_path, '{}.png'.format(frame_idx)) for frame_idx in range(center_frame_idx - margin, center_frame_idx + margin + 1)]
    frames = []
    for i in range(len(cfg.frame_extract_pattern)):
        if cfg.frame_extract_pattern[i]:
            frames.append(misc.imread(frames_path[i], mode = 'L'))

    frames_stack = np.asarray(frames)
    frames_stack = frames_stack.swapaxes(0,2)
    return [frames_stack, label]

class Data(RNGDataFlow):
    def __init__(self, train_or_test):
        assert train_or_test in ['train', 'test']
        fname_list = cfg.train_list if train_or_test == 'train' else cfg.test_list
        self.center_frame_list = []
        for fname in fname_list:
            self.center_frame_list.extend(get_center_frame_list(fname))
    
    def size(self):
        return int(len(self.center_frame_list) * 0.9)

    def get_data(self):
        idxs = np.arange(len(self.center_frame_list))
        #if self.shuffle:
        #    self.rng.shuffle(idxs)
        for k in idxs:
            frame_content = self.center_frame_list[k]
            res = read_data(frame_content)
            if res:
                yield res

if __name__ == '__main__':
    ds = Data('train')
    ds.reset_state()
    generator = ds.get_data()
    for i in generator:
        print(i[0].shape, i[1])
