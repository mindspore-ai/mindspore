# Copyright 2020 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from enum import IntEnum

class JointType(IntEnum):
    Nose = 0

    Neck = 1

    RightShoulder = 2

    RightElbow = 3

    RightHand = 4

    LeftShoulder = 5

    LeftElbow = 6

    LeftHand = 7

    RightWaist = 8

    RightKnee = 9

    RightFoot = 10

    LeftWaist = 11

    LeftKnee = 12

    LeftFoot = 13

    RightEye = 14

    LeftEye = 15

    RightEar = 16

    LeftEar = 17

params = {
    # paths
    'data_dir': './dataset',
    'save_model_path': './checkpoints/',
    'load_pretrain': False,
    'pretrained_model_path': "",

    # train type
    'train_type': 'fix_loss_scale', # chose in ['clip_grad', 'fix_loss_scale']
    'train_type_NP': 'clip_grad',

    # vgg bn
    'vgg_with_bn': False,
    'vgg_path': './vgg_model/vgg19-0-97_5004.ckpt',

    # if clip_grad
    'GRADIENT_CLIP_TYPE': 1,
    'GRADIENT_CLIP_VALUE': 10.0,

    # optimizer and lr
    'optimizer': "Adam", # chose in ['Momentum', 'Adam']
    'optimizer_NP': "Momentum",
    'group_params': True,
    'group_params_NP': False,
    'lr': 1e-4,
    'lr_type': 'default', # chose in ["default", "cosine"]
    'lr_gamma': 0.1,                    # if default
    'lr_steps': '100000,200000,250000', # if default
    'lr_steps_NP': '250000,300000',     # if default
    'warmup_epoch': 5,                  # if cosine
    'max_epoch_train': 60,
    'max_epoch_train_NP': 80,

    'loss_scale': 16384,

    # default param
    'batch_size': 10,
    'min_keypoints': 5,
    'min_area': 32 * 32,
    'insize': 368,
    'downscale': 8,
    'paf_sigma': 8,
    'heatmap_sigma': 7,
    'eva_num': 100,
    'keep_checkpoint_max': 1,
    'log_interval': 100,
    'ckpt_interval': 5304,

    'min_box_size': 64,
    'max_box_size': 512,
    'min_scale': 0.5,
    'max_scale': 2.0,
    'max_rotate_degree': 40,
    'center_perterb_max': 40,

    # inference params
    'inference_img_size': 368,
    'inference_scales': [0.5, 1, 1.5, 2],
    # 'inference_scales': [1.0],
    'heatmap_size': 320,
    'gaussian_sigma': 2.5,
    'ksize': 17,
    'n_integ_points': 10,
    'n_integ_points_thresh': 8,
    'heatmap_peak_thresh': 0.05,
    'inner_product_thresh': 0.05,
    'limb_length_ratio': 1.0,
    'length_penalty_value': 1,
    'n_subset_limbs_thresh': 3,
    'subset_score_thresh': 0.2,
    'limbs_point': [
        [JointType.Neck, JointType.RightWaist],
        [JointType.RightWaist, JointType.RightKnee],
        [JointType.RightKnee, JointType.RightFoot],
        [JointType.Neck, JointType.LeftWaist],
        [JointType.LeftWaist, JointType.LeftKnee],
        [JointType.LeftKnee, JointType.LeftFoot],
        [JointType.Neck, JointType.RightShoulder],
        [JointType.RightShoulder, JointType.RightElbow],
        [JointType.RightElbow, JointType.RightHand],
        [JointType.RightShoulder, JointType.RightEar],
        [JointType.Neck, JointType.LeftShoulder],
        [JointType.LeftShoulder, JointType.LeftElbow],
        [JointType.LeftElbow, JointType.LeftHand],
        [JointType.LeftShoulder, JointType.LeftEar],
        [JointType.Neck, JointType.Nose],
        [JointType.Nose, JointType.RightEye],
        [JointType.Nose, JointType.LeftEye],
        [JointType.RightEye, JointType.RightEar],
        [JointType.LeftEye, JointType.LeftEar]
    ],
    'joint_indices': [
        JointType.Nose,
        JointType.LeftEye,
        JointType.RightEye,
        JointType.LeftEar,
        JointType.RightEar,
        JointType.LeftShoulder,
        JointType.RightShoulder,
        JointType.LeftElbow,
        JointType.RightElbow,
        JointType.LeftHand,
        JointType.RightHand,
        JointType.LeftWaist,
        JointType.RightWaist,
        JointType.LeftKnee,
        JointType.RightKnee,
        JointType.LeftFoot,
        JointType.RightFoot
    ],

    # face params
    'face_inference_img_size': 368,
    'face_heatmap_peak_thresh': 0.1,
    'face_crop_scale': 1.5,
    'face_line_indices': [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], # 轮廓
        [17, 18], [18, 19], [19, 20], [20, 21],
        [22, 23], [23, 24], [24, 25], [25, 26],
        [27, 28], [28, 29], [29, 30],
        [31, 32], [32, 33], [33, 34], [34, 35],
        [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36],
        [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42],
        [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 48], # 唇外廓
        [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 60]
    ],

    # hand params
    'hand_inference_img_size': 368,
    'hand_heatmap_peak_thresh': 0.1,
    'fingers_indices': [
        [[0, 1], [1, 2], [2, 3], [3, 4]],
        [[0, 5], [5, 6], [6, 7], [7, 8]],
        [[0, 9], [9, 10], [10, 11], [11, 12]],
        [[0, 13], [13, 14], [14, 15], [15, 16]],
        [[0, 17], [17, 18], [18, 19], [19, 20]],
    ],
}
