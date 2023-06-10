# Copyright 2023 Huawei Technologies Co., Ltd
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
"""define constants"""
from tbe import tik

BLOCK_NUM = 16
FP16 = "float16"
INT8 = "int8"
INT32 = "int32"
FP32 = "float32"
REPEAT_SZ = 128
BLK_STRIDE = 1
REPEAT_STRIDE = 8
TRANS_CUBE_TGT = 8
FP16_MIN_VAL = -65504.0
MASK_FILL_VALUE = -10000.0
GM = tik.scope_gm
L1 = tik.scope_cbuf
L1OUT = tik.scope_cbuf_out
UB = tik.scope_ubuf
L0A = tik.scope_ca
L0B = tik.scope_cb
L0C = tik.scope_cc
DTYPE_SIZE = {
    "int8": 1,
    "float16": 2,
    "int16": 2,
    "float32": 4,
}
