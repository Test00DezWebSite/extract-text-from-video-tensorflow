from easydict import EasyDict as edict

cfg = edict()

cfg.train_list = ['data_train.txt']
cfg.test_list = ['data_test.txt']

cfg.h = 600
cfg.w = 600
cfg.max_deg = 20
cfg.class_num = 2

# Use a list to present a pattern indicating how to sample frames from video.
# e.g.: [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1] means that
#       take one frame and its blabla.
# Work is done in preprocess procedure so keep this parameter same as your data.
cfg.frame_extract_pattern = [1] * 11

cfg.with_bn = True
cfg.weight_decay = 5e-4

# depth of resnet, should be one of `{18, 34, 50, 101}`
cfg.depth = 18

cfg.batch_size = 8
cfg.learning_rate = [
    (1, 1e-1),
    (50, 3e-2),
    (100, 1e-2)
]
cfg.max_epoch = 150
