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
#include "nnacl/int8/matmul_int8.h"
#include "nnacl/errorcode.h"

size_t Conv1x1PackWeightSize(int32_t input_channel, int32_t output_channel, bool support_optimize) {
  size_t size = 0;
#ifdef ENABLE_ARM32
  size += UP_ROUND(input_channel, C16NUM) * UP_ROUND(output_channel, C2NUM) * sizeof(int8_t);
  size += (size_t)UP_ROUND(output_channel, C2NUM) * sizeof(int32_t);
  return size;
#else
  size += support_optimize ? UP_ROUND(input_channel, C4NUM) * UP_ROUND(output_channel, C16NUM) * sizeof(int8_t)
                           : UP_ROUND(input_channel, C16NUM) * UP_ROUND(output_channel, C4NUM) * sizeof(int8_t);
  size += support_optimize ? UP_ROUND(output_channel, C16NUM) * sizeof(int32_t)
                           : UP_ROUND(output_channel, C4NUM) * sizeof(int32_t);
#endif
  return size;
}

int Conv1x1Init(int8_t *src_weight, int32_t *src_bias, int32_t *filter_zps, int32_t input_channel,
                int32_t output_channel, int32_t input_zp, bool support_optimize, bool filter_peroc,
                int8_t **packed_weight, int32_t **bias_data, uint8_t *buf, size_t *offset, size_t buf_size) {
  if (packed_weight == NULL || bias_data == NULL) {
    return NNACL_ERR;
  }
#ifdef ENABLE_ARM32
  /* InitWeightBiasArm32 */
  /* weight */
  size_t size = UP_ROUND(input_channel, C16NUM) * UP_ROUND(output_channel, C2NUM) * sizeof(int8_t);
  if ((*offset + size) > buf_size) {
    return NNACL_ERR;
  }
  int8_t *packed_weight_ = (int8_t *)(buf + *offset);
  *offset += size;
  memset(packed_weight_, 0, size);
  RowMajor2Row2x16MajorInt8(src_weight, packed_weight_, output_channel, input_channel);
  /* bias */
  size = (size_t)UP_ROUND(output_channel, C2NUM);
  if ((*offset + size * sizeof(int32_t)) > buf_size) {
    return NNACL_ERR;
  }
  int32_t *bias_data_ = (int32_t *)(buf + *offset);
  *offset += size * sizeof(int32_t);
  if (bias_data_ == NULL) {
    free(packed_weight_);
    return NNACL_ERR;
  }
  memset(bias_data_, 0, size * sizeof(int32_t));
  if (src_bias != NULL) {
    memcpy(bias_data_, src_bias, (size_t)output_channel * sizeof(int32_t));
  }
#else
  /* InitWeightBias */
  /* weight */
  size_t size = support_optimize ? UP_ROUND(input_channel, C4NUM) * UP_ROUND(output_channel, C16NUM) * sizeof(int8_t)
                                 : UP_ROUND(input_channel, C16NUM) * UP_ROUND(output_channel, C4NUM) * sizeof(int8_t);
  int8_t *packed_weight_ = ((*offset + size) <= buf_size) ? (int8_t *)(buf + *offset) : NULL;
  if (packed_weight_ == NULL) {
    return NNACL_ERR;
  }
  *offset += size;
  memset(packed_weight_, 0, size);
  if (support_optimize) {
    RowMajor2Row4x16MajorInt8(src_weight, packed_weight_, output_channel, input_channel);
  } else {
    RowMajor2Row16x4MajorInt8(src_weight, packed_weight_, output_channel, input_channel);
  }
  /* bias */
  size = support_optimize ? UP_ROUND(output_channel, C16NUM) : UP_ROUND(output_channel, C4NUM);
  int32_t *bias_data_ = ((*offset + size * sizeof(int32_t)) <= buf_size) ? (int32_t *)(buf + *offset) : NULL;
  *offset += size * sizeof(int32_t);
  if (bias_data_ == NULL) {
    free(packed_weight_);
    packed_weight_ = NULL;
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
