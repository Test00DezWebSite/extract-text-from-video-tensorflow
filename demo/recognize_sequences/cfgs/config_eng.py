import numpy as np
from easydict import EasyDict as edict

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

cfg = edict()

cfg.name = 'english'

cfg.input_height = 20
cfg.input_width = None
cfg.input_channel = 1

cfg.cnn = edict()
cfg.cnn.padding = "SAME"
cfg.cnn.channels = [32, 32, 32, 32, 64, 64,64]
cfg.cnn.kernel_heights = [3, 3, 3, 3, 3, 3,3]
cfg.cnn.kernel_widths = [3, 3, 3, 3, 3, 3,3]
cfg.cnn.with_bn = True

cfg.rnn = edict()
cfg.rnn.hidden_size = 660
cfg.rnn.hidden_layers_no = 3

cfg.weight_decay = 5e-4

cfg.dictionary = [" ", "\"", "$", "%", "&", "'", "(", ")", "*",
                  "-", ".", "/", "0", "1", "2", "3", "4", "5",
                  "6", "7", "8", "9", ":", "<", ">", "?", "[",
                  "]", "a", "b", "c", "d", "e", "f", "g", "h",
                  "i", "j", "k", "l", "m", "n", "o", "p", "q",
                  "r", "s", "t", "u", "v", "w", "x", "y", "z",
                  "{", "}", 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                  'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                  'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
                  'Z', ';', ',', '!', '=']

cfg.label_size = len(cfg.dictionary) + 1


cfg.learning_rate = [(0, 1e-5), (3, 3e-5), (6, 6e-5), (10, 1e-4), (60, 1e-5)]

cfg.augmentors = [
            imgaug.ToFloat32(),
            imgaug.RandomOrderAug(
                [imgaug.Brightness(30, clip=False),
                 imgaug.Contrast((0.8, 1.2), clip=False)]),
            imgaug.Clip(),
            imgaug.ToUint8(),
        ]

cfg.train_list = ['' + "text_SeqRecog_train.txt"]
cfg.test_list = '' + "text_SeqRecog_test.txt"
