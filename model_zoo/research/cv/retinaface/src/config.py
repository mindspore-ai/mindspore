# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Config for train and eval."""
cfg_res50 = {
    'name': 'ResNet50',
    'device_target': "Ascend",
    'device_id': 0,
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'class_weight': 1.0,
    'landm_weight': 1.0,
    'batch_size': 8,
    'num_workers': 16,
    'num_anchor': 29126,
    'nnpu': 8,
    'image_size': 840,
    'in_channel': 256,
    'out_channel': 256,
    'match_thresh': 0.35,

    # opt
    'optim': 'sgd',  # 'sgd' or 'momentum'
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'loss_scale': 1,

    # seed
    'seed': 1,

    # lr
    'epoch': 60,
    'T_max': 50,  # cosine_annealing
    'eta_min': 0.0,  # cosine_annealing
    'decay1': 20,
    'decay2': 40,
    'lr_type': 'dynamic_lr',  # 'dynamic_lr' or cosine_annealing
    'initial_lr': 0.04,
    'warmup_epoch': -1,  # dynamic_lr: -1, cosine_annealing:0
    'gamma': 0.1,

    # checkpoint
    'ckpt_path': './checkpoint/',
    'keep_checkpoint_max': 8,
    'resume_net': None,

    # dataset
    'training_dataset': '../data/widerface/train/label.txt',
    'pretrain': True,
    'pretrain_path': '../data/resnet-90_625.ckpt',

    # val
    'val_model': './train_parallel3/checkpoint/ckpt_3/RetinaFace-56_201.ckpt',
    'val_dataset_folder': './data/widerface/val/',
    'val_origin_size': True,
    'val_confidence_threshold': 0.02,
    'val_nms_threshold': 0.4,
    'val_iou_threshold': 0.5,
    'val_save_result': False,
    'val_predict_save_folder': './widerface_result',
    'val_gt_dir': './data/ground_truth/',

    # infer
    'infer_dataset_folder': '/home/dataset/widerface/val/',
    'infer_gt_dir': '/home/dataset/widerface/ground_truth/',
}

cfg_mobile025 = {
    'name': 'MobileNet025',
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'class_weight': 1.0,
    'landm_weight': 1.0,
    'batch_size': 8,
    'num_workers': 12,
    'num_anchor': 16800,
    'ngpu': 2,
    'image_size': 640,
    'in_channel': 32,
    'out_channel': 64,
    'match_thresh': 0.35,

    # opt
    'optim': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,

    # seed
    'seed': 1,

    # lr
    'epoch': 120,
    'decay1': 70,
    'decay2': 90,
    'lr_type': 'dynamic_lr',
    'initial_lr': 0.02,
    'warmup_epoch': 5,
    'gamma': 0.1,

    # checkpoint
    'ckpt_path': './checkpoint/',
    'save_checkpoint_steps': 2000,
    'keep_checkpoint_max': 3,
    'resume_net': None,

    # dataset
    'training_dataset': '../data/widerface/train/label.txt',
    'pretrain': False,
    'pretrain_path': '../data/mobilenetv1-90_5004.ckpt',

    # val
    'val_model': './checkpoint/ckpt_0/RetinaFace-117_804.ckpt',
    'val_dataset_folder': './data/widerface/val/',
    'val_origin_size': False,
    'val_confidence_threshold': 0.02,
    'val_nms_threshold': 0.4,
    'val_iou_threshold': 0.5,
    'val_save_result': False,
    'val_predict_save_folder': './widerface_result',
    'val_gt_dir': './data/ground_truth/',
}
