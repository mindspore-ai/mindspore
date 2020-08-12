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
#ifndef MINDSPORE_LITE_SRC_BACKEND_ARM_NNACL_FP32_SPACE_TO_BATCH_H_
#define MINDSPORE_LITE_SRC_BACKEND_ARM_NNACL_FP32_SPACE_TO_BATCH_H_
#include "nnacl/op_base.h"

#define SPACE_TO_BATCH_BLOCK_SIZES_SIZE 2
#define SPACE_TO_BATCH_PADDINGS_SIZE 4

typedef struct SpaceToBatchParameter {
  OpParameter op_parameter_;
  int block_sizes_[8];
  int paddings_[8];
  int n_dims_;
  int num_elements_;
  int num_elements_padded_;
  int n_space_dims_;
  int in_shape_[8];
  int padded_in_shape_[8];
  bool need_paddings_;
} SpaceToBatchParameter;
#ifdef __cplusplus
extern "C" {
#endif
int SpaceToBatch(const float *input, float *output, SpaceToBatchParameter param, float *tmp_space[3]);
int SpaceToBatchForNHWC(const float *input, float *output, int *in_shape, int shape_size, int *block_size);
void TransposeForNHWC(const float *in_data, float *out_data, int *strides, int *out_strides, int *perm,
                      int *output_shape);
int EnumElement(int *shape, int n_dims);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_SRC_BACKEND_ARM_NNACL_FP32_SPACE_TO_BATCH_H_
