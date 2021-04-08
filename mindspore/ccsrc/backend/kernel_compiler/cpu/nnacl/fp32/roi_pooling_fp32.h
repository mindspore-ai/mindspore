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
#ifndef MINDSPORE_NNACL_FP32_ROI_POOLING_H_
#define MINDSPORE_NNACL_FP32_ROI_POOLING_H_

#include "nnacl/op_base.h"

typedef struct ROIPoolingParameter {
  // primitive parameter
  OpParameter op_parameter_;
  int pooledW_;
  int pooledH_;
  float scale_;

  // shape correlative
  int in_strides_[DIMENSION_4D];
  int out_strides_[DIMENSION_4D];
  int ndim_;
  int input_w_;
  int input_h_;
  int input_n_;
  int input_c_;
  int output_w_;
  int output_h_;
  int output_n_;
  int output_c_;

  // other parameter
  int thread_num_;
} ROIPoolingParameter;

#ifdef __cplusplus
extern "C" {
#endif
int ROIPooling(const float *in_ptr, float *out_ptr, const float *roi, float *max_c, int tid,
               const ROIPoolingParameter *param);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_ROI_POOLING_H_
