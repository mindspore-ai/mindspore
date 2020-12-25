# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from easydict import EasyDict as edict

config = edict()

# pose_resnet related params
POSE_RESNET = edict()
POSE_RESNET.NUM_LAYERS = 50
POSE_RESNET.DECONV_WITH_BIAS = False
POSE_RESNET.NUM_DECONV_LAYERS = 3
POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 256]
POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
POSE_RESNET.FINAL_CONV_KERNEL = 1
POSE_RESNET.TARGET_TYPE = 'gaussian'
POSE_RESNET.HEATMAP_SIZE = [48, 64]  # width * height, ex: 24 * 32
POSE_RESNET.SIGMA = 2

MODEL_EXTRAS = {
    'pose_resnet': POSE_RESNET,
}

# common params for NETWORK
config.MODEL = edict()
config.MODEL.NAME = 'pose_resnet'
config.MODEL.INIT_WEIGHTS = True
config.MODEL.PRETRAINED = './models/resnet50.ckpt'
config.MODEL.NUM_JOINTS = 17
config.MODEL.IMAGE_SIZE = [192, 256]  # width * height, ex: 192 * 256
config.MODEL.EXTRA = MODEL_EXTRAS[config.MODEL.NAME]

# dataset
config.DATASET = edict()
config.DATASET.ROOT = '/data/coco2017/'
config.DATASET.TEST_SET = 'val2017'
config.DATASET.TRAIN_SET = 'train2017'
# data augmentation
config.DATASET.FLIP = True
config.DATASET.ROT_FACTOR = 40
config.DATASET.SCALE_FACTOR = 0.3

# for train
config.TRAIN = edict()
config.TRAIN.BATCH_SIZE = 64
config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 140
config.TRAIN.LR = 0.001
config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 120]

# test
config.TEST = edict()
config.TEST.BATCH_SIZE = 32
config.TEST.FLIP_TEST = True
config.TEST.POST_PROCESS = True
config.TEST.SHIFT_HEATMAP = True
config.TEST.USE_GT_BBOX = False
config.TEST.MODEL_FILE = ''
config.TEST.COCO_BBOX_FILE = 'experiments/COCO_val2017_detections_AP_H_56_person.json'
# nms
config.TEST.OKS_THRE = 0.9
config.TEST.IN_VIS_THRE = 0.2
config.TEST.BBOX_THRE = 1.0
config.TEST.IMAGE_THRE = 0.0
config.TEST.NMS_THRE = 1.0
