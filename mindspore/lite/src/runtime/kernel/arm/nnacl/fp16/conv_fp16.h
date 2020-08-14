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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP16_CONV_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP16_CONV_FP16_H_

#include <arm_neon.h>
#include "nnacl/conv_parameter.h"

#ifndef ENABLE_NEON
void IndirectGemmFp16_16x8(float16_t *output, float16_t *input, float16_t *weight, float16_t *bias, size_t step,
                           size_t ic4, size_t oc8, size_t offset, size_t mode, size_t writeC4, size_t relu,
                           size_t relu6);
#endif

#ifdef __cplusplus
extern "C" {
#endif
void SWBorderFp16(float16_t *dst, const float16_t *src, const float16_t *weight, const float16_t *bias, int top,
                  int bottom, int left, int right, const ConvParameter *conv_param, const SlidingWindowParam *sliding);

void SWCenterFp16(float16_t *dst, const float16_t *src, const float16_t *weight, const float16_t *bias, int height,
                  int width, int kernel_h, int kernel_w, int out_h_step, int block_channel, int ic, int in_sh_step,
                  int in_sw_step, int in_kh_step, int in_kw_step, bool is_relu, bool is_relu6);

// fp16 sliding window
void ConvSWFp16(const float16_t *input_data, const float16_t *packed_weight, const float16_t *bias_data,
                float16_t *tmp_out_block, float16_t *output_data, int task_id, ConvParameter *conv_param,
                SlidingWindowParam *slidingWindow_param);

// fp16 convolution common (im2col+gemm)
void ConvFp16(float16_t *input_data, float16_t *packed_input, float16_t *packed_weight, float16_t *bias_data,
              float16_t *tmp_out_block, float16_t *output_data, int task_id, ConvParameter *conv_param);

// fp16 conv3x3
void Conv3x3Fp16(float16_t *input_data, float16_t *transed_weight, const float16_t *bias_data, float16_t *output_data,
                 float16_t *tile_buffer, float16_t *block_unit_buffer, float16_t *tmp_dst_buffer, float16_t *tmp_out,
                 int task_id, ConvParameter *conv_param);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP16_CONV_FP16_H_
