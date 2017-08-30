from easydict import EasyDict as edict

cfg = edict()

# ============ Path ============
cfg.train_list = 'train.txt'
cfg.test_list = 'test.txt'

# ============ Preprocess ============
cfg.h = 600
cfg.w = 600
cfg.max_deg = 20
cfg.class_num = 2

# ============ Graph ============
cfg.dilations = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16]
cfg.channels = [32, 32, 32, 32, 64, 64, 64, 64, 64]
cfg.kernel_size = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
cfg.with_bn = True
cfg.weight_decay = 5e-4

# ============ Train ============
cfg.batch_size = 8
cfg.learning_rate = [
    (1, 1e-1),
    (50, 3e-2),
    (100, 1e-2)
]
cfg.max_epoch = 150