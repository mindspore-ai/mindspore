/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_ERRORCODE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_ERRORCODE_H_

enum ErrorCodeCommonEnum {
  OPCLIB_OK = 0,
  OPCLIB_ERR = 1,
  OPCLIB_NULL_PTR,
  OPCLIB_PARAM_INVALID,
  OPLIB_COMMON_END = 9999
};

enum ErrorCodeFp32OpEnum {
  OPCLIB_ERRCODE_OP_FP32_START = 10000,
  OPCLIB_ERRCODE_STRASSEN_RECURSION_MALLOC,
  OPCLIB_ERRCODE_REVERSE_MALLOC,
  OPCLIB_ERRCODE_SQRT_NEGATIVE,
  OPCLIB_ERRCODE_RSQRT_NEGATIVE_OR_ZERO,
  OPCLIB_ERRCODE_LOG_NEGATIVE_OR_ZERO,
  OPCLIB_ERRCODE_DIVISOR_ZERO,
  OPCLIB_ERRCODE_INDEX_OUT_OF_RANGE,
  OPCLIB_ERRCODE_OP_FP32_END = 19999
};

enum ErrorCodeFp16OpEnum { OPCLIB_ERRCODE_OP_FP16_START = 20000, OPCLIB_ERRCODE_OP_FP16_END = 29999 };

enum ErrorCodeUint8OpEnum { OPCLIB_ERRCODE_OP_UINT8_START = 30000, OPCLIB_ERRCODE_OP_UINT8_END = 39999 };

enum ErrorCodeInt8OpEnum { OPCLIB_ERRCODE_OP_INT8_START = 40000, OPCLIB_ERRCODE_OP_INT8_END = 49999 };

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_ERRORCODE_H_

