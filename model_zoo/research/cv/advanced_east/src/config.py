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
"""
configs
"""
from easydict import EasyDict

config = EasyDict({
    'epoch_num': 6,
    'learning_rate_ascend': 1e-4,
    'learning_rate_gpu': 1e-3,
    'decay_ascend': 3e-5,
    'decay_gpu': 5e-4,
    'epsilon': 1e-4,
    'batch_size': 2,
    'ckpt_interval': 2,
    'lambda_inside_score_loss': 4.0,
    'lambda_side_vertex_code_loss': 1.0,
    "lambda_side_vertex_coord_loss": 1.0,
    'max_train_img_size': 448,
    'max_predict_img_size': 448,
    'train_img_size': [256, 384, 448, 512, 640, 736, 768],
    'predict_img_size': [256, 384, 448, 512, 640, 736, 768],
    'ckpt_save_max': 500,
    'validation_split_ratio': 0.1,
    'total_img': 10000,
    'data_dir': './icpr/',
    'train_fname': 'train.txt',
    'train_fname_var': 'train_',
    'val_fname': 'val.txt',
    'val_fname_var': 'val_',
    'mindsrecord_train_file': 'advanced-east.mindrecord',
    'mindsrecord_test_file': 'advanced-east-val.mindrecord',
    'results_dir': './results/',
    'origin_image_dir_name': 'images/',
    'train_image_dir_name': 'images_train/',
    'origin_txt_dir_name': 'txt/',
    'train_label_dir_name': 'labels_train/',
    'train_image_dir_name_var': 'images_train_',
    'mindsrecord_train_file_var': 'advanced-east_',
    'train_label_dir_name_var': 'labels_train_',
    'mindsrecord_val_file_var': 'advanced-east-val_',
    'show_gt_image_dir_name': 'show_gt_images/',
    'show_act_image_dir_name': 'show_act_images/',
    'saved_model_file_path': './saved_model/',
    'last_model_name': '_.ckpt',
    'pixel_threshold': 0.8,
    'iou_threshold': 0.1,
    'feature_layers_range': range(5, 1, -1),
    'feature_layers_num': len(range(5, 1, -1)),
    'pixel_size': 2 ** range(5, 1, -1)[-1],
    'quiet': True,
    'side_vertex_pixel_threshold': 0.8,
    'trunc_threshold': 0.1,
    'predict_cut_text_line': False,
    'predict_write2txt': True,
    'shrink_ratio': 0.2,
    'shrink_side_ratio': 0.6,
    'gen_origin_img': True,
    'draw_gt_quad': False,
    'draw_act_quad': False,
    'vgg_npy': './vgg16.npy',
    'vgg_weights': './src/0-150_5004.ckpt',
    'ds_sink_mode': False
})

cifar_cfg = EasyDict({
    "num_classes": 10,
    "lr": 0.01,
    "lr_init": 0.01,
    "lr_max": 0.1,
    "lr_epochs": '30,60,90,120',
    "lr_scheduler": "step",
    "warmup_epochs": 5,
    "batch_size": 64,
    "max_epoch": 70,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "loss_scale": 1.0,
    "label_smooth": 0,
    "label_smooth_factor": 0,
    "buffer_size": 10,
    "image_size": '224,224',
    "pad_mode": 'same',
    "padding": 0,
    "has_bias": False,
    "batch_norm": True,
    "keep_checkpoint_max": 10,
    "initialize_mode": "XavierUniform",
    "has_dropout": False
})
