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

#include "wrapper/fp32/deconvolution_fp32_wrapper.h"
#include "nnacl/fp32/deconv_fp32.h"
#include "nnacl/fp32/matmul_fp32.h"

int DoDeconvFp32(const float *packed_input, const float *packed_weight, const float *packed_bias, float *packed_output,
                 float *output, float *tmp_ori_buffer, const MatMulParameter *matmul_param,
                 const ConvParameter *conv_param, int task_id) {
  int thread_count = MSMIN(conv_param->thread_num_, UP_DIV(conv_param->output_channel_, C8NUM));
  int thread_stride = UP_DIV(UP_DIV(conv_param->output_channel_, C8NUM), thread_count);
  int res_stride = UP_DIV(conv_param->output_channel_, C8NUM) - task_id * thread_stride;
  int oc = MSMIN(thread_stride, res_stride);
  int cur_stride = thread_stride * C8NUM;
  res_stride = conv_param->output_channel_ - task_id * thread_stride * C8NUM;
  int oc_res = MSMIN(cur_stride, res_stride);
  if (oc <= 0 || oc_res <= 0) {
    return NNACL_OK;
  }

  int kernel_plane = conv_param->kernel_w_ * conv_param->kernel_h_;
  int output_plane = conv_param->output_h_ * conv_param->output_w_;

#if defined(ENABLE_ARM32)
  float *tmp_buffer = tmp_ori_buffer + task_id * thread_stride * C8NUM * kernel_plane * matmul_param->row_4_;
  MatMulOpt(packed_input, packed_weight + task_id * thread_stride * C8NUM * kernel_plane * matmul_param->deep_,
            tmp_buffer, NULL, ActType_No, matmul_param->deep_, matmul_param->row_4_, oc * C8NUM * kernel_plane,
            matmul_param->col_, OutType_C8);
#else
  float *tmp_buffer = tmp_ori_buffer + task_id * thread_stride * C8NUM * kernel_plane * matmul_param->row_12_;
  MatMulOpt(packed_input, packed_weight + task_id * thread_stride * C8NUM * kernel_plane * matmul_param->deep_,
            tmp_buffer, NULL, ActType_No, matmul_param->deep_, matmul_param->row_12_, oc * C8NUM * kernel_plane,
            matmul_param->col_, OutType_C8);
#endif

  DeConvPostFp32C8(tmp_buffer, packed_output + task_id * thread_stride * C8NUM * output_plane,
                   packed_bias + thread_stride * task_id * C8NUM, output + task_id * thread_stride * C8NUM, oc_res,
                   conv_param);
  return NNACL_OK;
}

int DeConvFp32Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  DeConvFp32Args *args = (DeConvFp32Args *)cdata;
  const MatMulParameter *matmul_param = args->matmul_param_;
  const ConvParameter *conv_param = args->conv_param_;
  const float *packed_input = args->packed_input_;
  const float *packed_weight = args->packed_weight_;
  const float *packed_bias = args->packed_bias_;
  float *packed_output = args->packed_output_;
  float *output = args->output_;
  float *tmp_buffer = args->tmp_buffer_;
  DoDeconvFp32(packed_input, packed_weight, packed_bias, packed_output, output, tmp_buffer, matmul_param, conv_param,
               task_id);
  return NNACL_OK;
}
