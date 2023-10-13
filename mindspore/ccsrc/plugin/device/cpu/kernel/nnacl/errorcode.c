/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "nnacl/errorcode.h"
#include <stdbool.h>

void InitNNACLKernelErrorCode(char **nnacl_kernel_error_msg) {
  nnacl_kernel_error_msg[NNACL_CROP_AND_RESIZE_BOX_IDX_INVALID] =
    "In CropAndResize, the value of box idx should match: [0, batch).";
  nnacl_kernel_error_msg[NNACL_WHERE_INPUT_NUM_INVALID] = "Invalid input number. Where op input number support 1 or 3.";
  nnacl_kernel_error_msg[NNACL_WHERE_CONDITION_DATA_TYPE_ERROR] =
    "Invalid input data type. Where op input data type support int32 fp32 and bool.";
  nnacl_kernel_error_msg[NNACL_WHERE_CONDITION_NUM_INVALID] =
    "The length of three inputs are not equal to 1 or length of output, which is unacceptable.";
  nnacl_kernel_error_msg[NNACL_WHERE_INVALID_OUT_NUM] = "The element number invalid.";
  nnacl_kernel_error_msg[NNACL_WHERE_NUM_MAX_INVALID] = "Inputs' length are zero";
  nnacl_kernel_error_msg[NNACL_ERR] = "NNACL common error.";
}

char *NNACLErrorMsg(int error_code) {
  static char nnacl_kernel_error_msg[NNACL_COMMON_END][MAX_MSG_LEN];
  static bool inited = false;
  if (!inited) {
    inited = true;
    InitNNACLKernelErrorCode((char **)nnacl_kernel_error_msg);
  }

  if (error_code > NNACL_OK && error_code < NNACL_COMMON_END) {
    return nnacl_kernel_error_msg[error_code];
  }

  return "NNACL execute error!";
}
