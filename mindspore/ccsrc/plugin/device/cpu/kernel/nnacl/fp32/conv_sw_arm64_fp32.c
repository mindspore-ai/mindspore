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

#include "nnacl/fp32/conv_sw_arm64_fp32.h"
#include "nnacl/fp32/conv_sw.h"

bool CheckArm64UseSWConv(const ConvParameter *conv_param) {
  if (conv_param->kernel_h_ == 1 && conv_param->kernel_w_ == 1) {
    return false;
  }
  if (conv_param->input_channel_ > C128NUM) {
    return false;
  }
  if (conv_param->kernel_h_ > C5NUM || conv_param->kernel_w_ > C5NUM) {
    return false;
  }
  if (conv_param->dilation_h_ != 1 || conv_param->dilation_w_ != 1) {
    return false;
  }
  if (conv_param->stride_w_ > C3NUM) {
    return false;
  }
  if (conv_param->input_h_ / conv_param->kernel_h_ < C48NUM || conv_param->input_w_ / conv_param->kernel_w_ < C48NUM) {
    return false;
  }
  return true;
}

typedef void (*SWConvKernel)(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                             size_t kernel_w, size_t act_flag, size_t oc_algin, size_t ic_algin, size_t in_kw_step,
                             size_t in_kh_step, size_t in_sw_step, size_t kw_remainder, size_t write_mode);

void SWConv1x8Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                     size_t kernel_w, size_t act_flag, size_t oc_algin, size_t ic_algin, size_t in_kw_step,
                     size_t in_kh_step, size_t in_sw_step, size_t kw_remainder, size_t write_mode);

void SWConv1x16Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                      size_t kernel_w, size_t act_flag, size_t oc_algin, size_t ic_algin, size_t in_kw_step,
                      size_t in_kh_step, size_t in_sw_step, size_t kw_remainder, size_t write_mode);

void SWConv2x8Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                     size_t kernel_w, size_t act_flag, size_t oc_algin, size_t ic_algin, size_t in_kw_step,
                     size_t in_kh_step, size_t in_sw_step, size_t kw_remainder, size_t write_mode);

void SWConv2x16Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                      size_t kernel_w, size_t act_flag, size_t oc_algin, size_t ic_algin, size_t in_kw_step,
                      size_t in_kh_step, size_t in_sw_step, size_t kw_remainder, size_t write_mode);

void SWConv3x8Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                     size_t kernel_w, size_t act_flag, size_t oc_algin, size_t ic_algin, size_t in_kw_step,
                     size_t in_kh_step, size_t in_sw_step, size_t kw_remainder, size_t write_mode);

void SWConv3x16Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                      size_t kernel_w, size_t act_flag, size_t oc_algin, size_t ic_algin, size_t in_kw_step,
                      size_t in_kh_step, size_t in_sw_step, size_t kw_remainder, size_t write_mode);

void SWConv4x8Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                     size_t kernel_w, size_t act_flag, size_t oc_algin, size_t ic_algin, size_t in_kw_step,
                     size_t in_kh_step, size_t in_sw_step, size_t kw_remainder, size_t write_mode);

void SWConv4x16Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                      size_t kernel_w, size_t act_flag, size_t oc_algin, size_t ic_algin, size_t in_kw_step,
                      size_t in_kh_step, size_t in_sw_step, size_t kw_remainder, size_t write_mode);

void SWConv5x8Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                     size_t kernel_w, size_t act_flag, size_t oc_algin, size_t ic_algin, size_t in_kw_step,
                     size_t in_kh_step, size_t in_sw_step, size_t kw_remainder, size_t write_mode);

void SWConv5x16Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                      size_t kernel_w, size_t act_flag, size_t oc_algin, size_t ic_algin, size_t in_kw_step,
                      size_t in_kh_step, size_t in_sw_step, size_t kw_remainder, size_t write_mode);

#define ROW_NUM_LIST const int ow_block_num[2] = {5, 5};
#define KERNEL_LIST                                                                        \
  const SWConvKernel kernel[2][5] = {                                                      \
    {SWConv1x8Kernel, SWConv2x8Kernel, SWConv3x8Kernel, SWConv4x8Kernel, SWConv5x8Kernel}, \
    {SWConv1x16Kernel, SWConv2x16Kernel, SWConv3x16Kernel, SWConv4x16Kernel, SWConv5x16Kernel}};
#define COMPUTE_CORE                                                                                              \
  kernel[oc_block - 1][ow_block - 1](dst_oc + ow * out_w_step, src_w, weight, bias, kernel_h, kernel_w, act_type, \
                                     out_block_step, ic_algin, in_kw_step, in_kh_step, in_sw_step, 0, write_mode);
#define OUTER_COMPUTE                                                                                                 \
  kernel(dst_kernel, src_kernel, weight_kernel, bias, end_kh - start_kh, end_kw - start_kw, act_type,                 \
         sw_param->out_block_step_, sw_param->ic_align_, sw_param->in_kw_step_, sw_param->in_kh_step_,                \
         sw_param->in_sw_step_, (conv_param->kernel_w_ - end_kw + start_kw) * C8NUM * oc_block * sw_param->ic_align_, \
         write_mode);
GenerateConvSWFunc(Arm64, C2NUM, ROW_NUM_LIST, KERNEL_LIST, COMPUTE_CORE, OUTER_COMPUTE);
