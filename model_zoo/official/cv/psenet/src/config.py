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


from easydict import EasyDict as ed

config = ed({
    "INFER_LONG_SIZE": 1920,
    "KERNEL_NUM": 7,
    "INFERENCE": True,  # INFER MODE\TRAIN MODE

    # backbone
    "BACKBONE_LAYER_NUMS": [3, 4, 6, 3],
    "BACKBONE_IN_CHANNELS": [64, 256, 512, 1024],
    "BACKBONE_OUT_CHANNELS": [256, 512, 1024, 2048],

    # neck
    "NECK_OUT_CHANNEL": 256,

    # lr
    "BASE_LR": 2e-3,
    "TRAIN_TOTAL_ITER": 58000,
    "WARMUP_STEP": 620,
    "WARMUP_RATIO": 1/3,

    # dataset for train
    "TRAIN_ROOT_DIR": "psenet/ic15/",
    "TRAIN_LONG_SIZE": 640,
    "TRAIN_MIN_SCALE": 0.4,
    "TRAIN_BATCH_SIZE": 4,
    "TRAIN_REPEAT_NUM": 1800,
    "TRAIN_DROP_REMAINDER": True,
    "TRAIN_MODEL_SAVE_PATH": "./checkpoints/",

    # dataset for test
    "TEST_ROOT_DIR": "psenet/ic15/",
    "TEST_BUFFER_SIZE": 4,
    "TEST_DROP_REMAINDER": False,
})
