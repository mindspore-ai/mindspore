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

#include "nnacl/fp16/common_func_fp16.h"

void PostConvFuncCommFp16(float16_t *out_ptr, const float16_t *src_ptr_, const float16_t *bias_ptr,
                          size_t output_channel, size_t plane_size, size_t oc_stride, size_t hw_stride,
                          ActType act_type, int size) {
  if (size == 0) {
    return;
  }
  for (int oc = 0; oc < output_channel; oc++) {
    int oc_div = oc / size, oc_mod = oc % size;
    for (int hw = 0; hw < plane_size; hw++) {
      int src_index = oc_div * size * hw_stride + hw * size + oc_mod;
      int dst_index = hw * oc_stride + oc;
      float16_t value = src_ptr_[src_index];
      if (bias_ptr != NULL) {
        value = value + bias_ptr[oc];
      }
      value = (act_type == ActType_Relu || act_type == ActType_Relu6) ? (MSMAX(0.f, value)) : (value);
      value = (act_type == ActType_Relu6) ? (MSMIN(6.f, value)) : (value);
      out_ptr[dst_index] = value;
    }
  }
  return;
}

void PostConvFuncFp16C8(const float16_t *c8_out, float16_t *nhwc_out, const float16_t *bias, size_t oc, size_t plane,
                        size_t oc_stride, ActType act_type) {
#ifdef ENABLE_ARM64
  size_t oc8mod = oc % C8NUM;
  size_t oc8div = oc - oc8mod;
  size_t stride_size = oc_stride * sizeof(float16_t);
  PostFuncBiasReluC8Fp16(nhwc_out, c8_out, bias, oc8div, oc8mod, plane, stride_size, act_type);
#else
  PostConvFuncCommFp16(nhwc_out, c8_out, bias, oc, plane, oc_stride, plane, act_type, C8NUM);
#endif
}

void PostConvFuncFp16C4(const float16_t *c4_out, float16_t *nhwc_out, const float16_t *bias, size_t oc, size_t plane,
                        size_t plane_stride, ActType act_type) {
#ifdef ENABLE_ARM64
  size_t oc4mod = oc % C4NUM;
  size_t oc4div = oc - oc4mod;
  size_t stride_size = (plane_stride - plane) * C4NUM * sizeof(float16_t);
  PostFuncBiasReluC4Fp16(nhwc_out, c4_out, bias, oc4div, oc4mod, plane, stride_size, act_type);
#else
  PostConvFuncCommFp16(nhwc_out, c4_out, bias, oc, plane, oc, plane_stride, act_type, C4NUM);
#endif
}

#ifdef ENABLE_ARM82_A32
void PostFuncBiasReluC4Fp16(float16_t *dst, const float16_t *src, const float16_t *bias, size_t oc4div, size_t oc4mod,
                            size_t plane_size, size_t plane_stride, size_t relu_type) {
  // TODO(fun): function
}

void PostFuncBiasReluC8Fp16(float16_t *dst, const float16_t *src, const float16_t *bias, size_t oc8div, size_t oc8mod,
                            size_t plane_size, size_t stride, size_t relu_type) {
  // TODO(fun): function
}
#endif
