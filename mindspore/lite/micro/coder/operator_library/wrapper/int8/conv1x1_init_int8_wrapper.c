/*
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "wrapper/int8/conv1x1_init_int8_wrapper.h"
#include <memory.h>
#include "nnacl/int8/matmul_int8.h"
#include "nnacl/errorcode.h"

int Conv1x1Init(int8_t *src_weight, int32_t *src_bias, int32_t *filter_zps, int32_t input_channel,
                int32_t output_channel, int32_t input_zp, bool support_optimize, bool filter_peroc,
                int8_t **packed_weight, int32_t **bias_data) {
  if (packed_weight == NULL || bias_data == NULL) {
    return NNACL_ERR;
  }
#ifdef ENABLE_ARM32
  /* InitWeightBiasArm32 */
  /* weight */
  size_t size = UP_ROUND(input_channel, C16NUM) * UP_ROUND(output_channel, C2NUM) * sizeof(int8_t);
  int8_t *packed_weight_ = (int8_t *)(malloc(size));
  if (packed_weight_ == NULL) {
    return NNACL_ERR;
  }
  memset(packed_weight_, 0, size);
  RowMajor2Row2x16MajorInt8(src_weight, packed_weight_, output_channel, input_channel);
  /* bias */
  size = UP_ROUND(output_channel, C2NUM);
  int32_t *bias_data_ = (int32_t *)malloc(size * sizeof(int32_t));
  if (bias_data_ == NULL) {
    free(packed_weight_);
    return NNACL_ERR;
  }
  memset(bias_data_, 0, size * sizeof(int32_t));
  if (src_bias != NULL) {
    memcpy(bias_data_, src_bias, output_channel * sizeof(int32_t));
  }
#else
  /* InitWeightBias */
  /* weight */
  size_t size = support_optimize ? UP_ROUND(input_channel, C4NUM) * UP_ROUND(output_channel, C16NUM) * sizeof(int8_t)
                                 : UP_ROUND(input_channel, C16NUM) * UP_ROUND(output_channel, C4NUM) * sizeof(int8_t);
  int8_t *packed_weight_ = (int8_t *)(malloc(size));
  if (packed_weight_ == NULL) {
    return NNACL_ERR;
  }
  memset(packed_weight_, 0, size);
  if (support_optimize) {
    RowMajor2Row4x16MajorInt8(src_weight, packed_weight_, output_channel, input_channel);
  } else {
    RowMajor2Row16x4MajorInt8(src_weight, packed_weight_, output_channel, input_channel);
  }
  /* bias */
  size = support_optimize ? UP_ROUND(output_channel, C16NUM) : UP_ROUND(output_channel, C4NUM);
  int32_t *bias_data_ = (int32_t *)malloc(size * sizeof(int32_t));
  if (bias_data_ == NULL) {
    free(packed_weight_);
    return NNACL_ERR;
  }
  memset(bias_data_, 0, size * sizeof(int32_t));
  if (src_bias != NULL) {
    memcpy(bias_data_, src_bias, output_channel * sizeof(int32_t));
  }
#endif
  /* InitBiasByzp */
  /* bias = bias - v2 x zp1 + zp1 x zp2  */
  for (int oc = 0; oc < output_channel; oc++) {
    int32_t weight_sum_value = 0;
    int32_t filter_zp = (filter_peroc) ? filter_zps[oc] : filter_zps[0];
    for (int ic = 0; ic < input_channel; ic++) {
      weight_sum_value += src_weight[oc * input_channel + ic];
    }
    bias_data_[oc] += filter_zp * input_zp * input_channel - weight_sum_value * input_zp;
  }

  *packed_weight = packed_weight_;
  *bias_data = bias_data_;
  return NNACL_OK;
}
