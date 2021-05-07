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

cfg_unet_medical = {
    'model': 'unet_medical',
    'crop': [388 / 572, 388 / 572],
    'img_size': [572, 572],
    'lr': 0.0001,
    'epochs': 400,
    'repeat': 400,
    'distribute_epochs': 1600,
    'batchsize': 16,
    'cross_valid_ind': 1,
    'num_classes': 2,
    'num_channels': 1,

    'keep_checkpoint_max': 10,
    'weight_decay': 0.0005,
    'loss_scale': 1024.0,
    'FixedLossScaleManager': 1024.0,

    'resume': False,
    'resume_ckpt': './',
    'transfer_training': False,
    'filter_weight': ['outc.weight', 'outc.bias'],
    'eval_activate': 'Softmax',
    'eval_resize': False
}

cfg_unet_nested = {
    'model': 'unet_nested',
    'crop': None,
    'img_size': [576, 576],
    'lr': 0.0001,
    'epochs': 400,
    'repeat': 400,
    'distribute_epochs': 1600,
    'batchsize': 16,
    'cross_valid_ind': 1,
    'num_classes': 2,
    'num_channels': 1,

    'keep_checkpoint_max': 10,
    'weight_decay': 0.0005,
    'loss_scale': 1024.0,
    'FixedLossScaleManager': 1024.0,
    'use_bn': True,
    'use_ds': True,
    'use_deconv': True,

    'resume': False,
    'resume_ckpt': './',
    'transfer_training': False,
    'filter_weight': ['final1.weight', 'final2.weight', 'final3.weight', 'final4.weight'],
    'eval_activate': 'Softmax',
    'eval_resize': False
}

cfg_unet_nested_cell = {
    'model': 'unet_nested',
    'dataset': 'Cell_nuclei',
    'crop': None,
    'img_size': [96, 96],
    'lr': 3e-4,
    'epochs': 200,
    'repeat': 10,
    'distribute_epochs': 1600,
    'batchsize': 16,
    'cross_valid_ind': 1,
    'num_classes': 2,
    'num_channels': 3,

    'keep_checkpoint_max': 10,
    'weight_decay': 0.0005,
    'loss_scale': 1024.0,
    'FixedLossScaleManager': 1024.0,
    'use_bn': True,
    'use_ds': True,
    'use_deconv': True,

    'resume': False,
    'resume_ckpt': './',
    'transfer_training': False,
    'filter_weight': ['final1.weight', 'final2.weight', 'final3.weight', 'final4.weight'],
    'eval_activate': 'Softmax',
    'eval_resize': False
}

cfg_unet_simple = {
    'model': 'unet_simple',
    'crop': None,
    'img_size': [576, 576],
    'lr': 0.0001,
    'epochs': 400,
    'repeat': 400,
    'distribute_epochs': 1600,
    'batchsize': 16,
    'cross_valid_ind': 1,
    'num_classes': 2,
    'num_channels': 1,

    'keep_checkpoint_max': 10,
    'weight_decay': 0.0005,
    'loss_scale': 1024.0,
    'FixedLossScaleManager': 1024.0,

    'resume': False,
    'resume_ckpt': './',
    'transfer_training': False,
    'filter_weight': ["final.weight"],
    'eval_activate': 'Softmax',
    'eval_resize': False
}

cfg_unet_simple_coco = {
    'model': 'unet_simple',
    'dataset': 'COCO',
    'split': 1.0,
    'img_size': [512, 512],
    'lr': 3e-4,
    'epochs': 80,
    'repeat': 1,
    'distribute_epochs': 120,
    'cross_valid_ind': 1,
    'batchsize': 16,
    'num_channels': 3,

    'keep_checkpoint_max': 10,
    'weight_decay': 0.0005,
    'loss_scale': 1024.0,
    'FixedLossScaleManager': 1024.0,

    'resume': False,
    'resume_ckpt': './',
    'transfer_training': False,
    'filter_weight': ["final.weight"],
    'eval_activate': 'Softmax',
    'eval_resize': False,

    'num_classes': 81,
    'coco_classes': ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                     'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                     'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                     'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                     'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                     'kite', 'baseball bat', 'baseball glove', 'skateboard',
                     'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                     'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                     'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                     'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                     'refrigerator', 'book', 'clock', 'vase', 'scissors',
                     'teddy bear', 'hair drier', 'toothbrush'),
    # change the following settings to real path
    'anno_json': '/data/coco2017/annotations/instances_train2017.json',
    'val_anno_json': '/data/coco2017/annotations/instances_val2017.json',
    'coco_dir': '/data/coco2017/train2017',
    'val_coco_dir': '/data/coco2017/val2017'
}

cfg_unet = cfg_unet_simple
if not ('dataset' in cfg_unet and cfg_unet['dataset'] == 'Cell_nuclei') and cfg_unet['eval_resize']:
    print("ISBI dataset not support resize to original image size when in evaluation.")
    cfg_unet['eval_resize'] = False
