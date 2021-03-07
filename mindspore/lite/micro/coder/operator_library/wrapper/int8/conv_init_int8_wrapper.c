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

#include "wrapper/int8/conv_init_int8_wrapper.h"
#include <memory.h>
#include "nnacl/op_base.h"
#include "nnacl/int8/matmul_int8.h"
#include "nnacl/errorcode.h"

int ConvInit(int8_t *origin_weight, const int32_t *ori_bias, const int32_t *filter_quant_zps, int kernel_h,
             int kernel_w, int input_channel, int output_channel, int32_t input_zp, bool filter_peroc,
             bool support_optimize, int8_t **packed_weight, int32_t **bias_data) {
  int8_t *packed_weight_ = NULL;
  int32_t *bias_data_ = NULL;
  int kernel_plane = kernel_h * kernel_w;
  int up_round_deep;
  int up_round_oc;
#ifdef ENABLE_ARM32
  up_round_oc = UP_ROUND(output_channel, C2NUM);
  up_round_deep = UP_ROUND(kernel_plane * input_channel, C16NUM);
#else
  if (support_optimize) {
    up_round_oc = UP_ROUND(output_channel, C8NUM);
    up_round_deep = UP_ROUND(kernel_plane * input_channel, C4NUM);
  } else {
    up_round_oc = UP_ROUND(output_channel, C4NUM);
    up_round_deep = UP_ROUND(kernel_plane * input_channel, C16NUM);
  }
#endif
  int pack_weight_size = up_round_oc * up_round_deep;
  size_t bias_size = up_round_oc * sizeof(int32_t);

  // init weight
  packed_weight_ = (int8_t *)(malloc(pack_weight_size));
  if (packed_weight_ == NULL) {
    return NNACL_ERR;
  }
  memset(packed_weight_, 0, pack_weight_size);
#ifdef ENABLE_ARM32
  RowMajor2Row2x16MajorInt8(origin_weight, packed_weight_, output_channel, input_channel * kernel_plane);
#else
  if (support_optimize) {
    RowMajor2Row8x4MajorInt8(origin_weight, packed_weight_, output_channel, input_channel * kernel_plane);
  } else {
    RowMajor2Row16x4MajorInt8(origin_weight, packed_weight_, output_channel, input_channel * kernel_plane);
  }
#endif

  // init bias
  bias_data_ = (int32_t *)(malloc(bias_size));
  if (bias_data_ == NULL) {
    free(packed_weight_);
    return NNACL_ERR;
  }
  memset(bias_data_, 0, bias_size);
  if (ori_bias != NULL) {
    memcpy(bias_data_, ori_bias, output_channel * sizeof(int32_t));
  }

  for (int oc = 0; oc < output_channel; oc++) {
    int32_t filter_zp = filter_quant_zps[0];
    if (filter_peroc) {
      filter_zp = filter_quant_zps[oc];
    }
    int32_t weight_sum_value = up_round_deep * filter_zp;
    for (int i = 0; i < kernel_plane * input_channel; i++) {
      weight_sum_value += origin_weight[oc * kernel_plane * input_channel + i] - filter_zp;
    }
    bias_data_[oc] += filter_zp * input_zp * up_round_deep - weight_sum_value * input_zp;
  }

  *packed_weight = packed_weight_;
  *bias_data = bias_data_;
  return NNACL_OK;
}
