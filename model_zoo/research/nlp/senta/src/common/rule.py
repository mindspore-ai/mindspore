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
some rule
"""


class MaxTruncation():
    """MaxTruncation：
    """
    KEEP_HEAD = 0
    KEEP_TAIL = 1
    KEEP_BOTH_HEAD_TAIL = 2


class InstanceName():
    """InstanceName:
    """
    RECORD_ID = "id"
    RECORD_EMB = "emb"
    SRC_IDS = "src_ids"
    MASK_IDS = "mask_ids"
    SEQ_LENS = "seq_lens"
    SENTENCE_IDS = "sent_ids"
    POS_IDS = "pos_ids"
    TASK_IDS = "task_ids"

    TRAIN_LABEL_SRC_IDS = "train_label_src_ids"
    TRAIN_LABEL_MASK_IDS = "train_label_mask_ids"
    TRAIN_LABEL_SEQ_LENS = "train_label_seq_lens"
    INFER_LABEL_SRC_IDS = "infer_label_src_ids"
    INFER_LABEL_MASK_IDS = "infer_label_mask_ids"
    INFER_LABEL_SEQ_LENS = "infer_label_seq_lens"

    SEQUENCE_EMB = "sequence_output"
    POOLED_EMB = "pooled_output"

    TARGET_FEED_NAMES = "target_feed_name"
    TARGET_PREDICTS = "target_predicts"
    PREDICT_RESULT = "predict_result"
    LABEL = "label"  # label
    LOSS = "loss"  # loss
    # CRF_EMISSION = "crf_emission"  # crf_emission

    TRAINING = "training"
    EVALUATE = "evaluate"
    TEST = "test"
    SAVE_INFERENCE = "save_inference"

    STEP = "steps"
    SPEED = "speed"
    TIME_COST = "time_cost"
    GPU_ID = "gpu_id"


class FieldLength():
    """
    FieldLength
    """
    CUSTOM_TEXT_FIELD = 3
    ERNIE_TEXT_FIELD = 6
    SINGLE_SCALAR_FIELD = 1
    ARRAY_SCALAR_FIELD = 2
    BASIC_TEXT_FIELD = 2
    GENERATE_LABEL_FIELD = 6
