/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "nnacl/fp32_grad/maxpool_grad_grad.h"
#include "nnacl/errorcode.h"

int MaxPoolGradGrad(const float *input, const float *grad, float *output, size_t start, size_t end,
                    PoolingParameter *param, PoolingComputeParam *args) {
  const int channel = args->input_channel_;
  const int input_height = args->input_h_;
  const int input_width = args->input_w_;

  const int window_height = args->window_h_;
  const int window_width = args->window_w_;

  const int stride_height = param->stride_h_;
  const int stride_width = param->stride_w_;

  const int pad_top = param->pad_u_;
  const int pad_left = param->pad_l_;

  const int output_height = args->output_h_;
  NNACL_CHECK_ZERO_RETURN_ERR(output_height);
  const int output_width = args->output_w_;
  NNACL_CHECK_ZERO_RETURN_ERR(output_width);

  const int output_chw = channel * output_height * output_width;
  NNACL_CHECK_ZERO_RETURN_ERR(output_chw);
  const int output_hw = output_height * output_width;
  NNACL_CHECK_ZERO_RETURN_ERR(output_hw);

  for (size_t pos = start; pos < end; pos++) {
    const int pos_n = pos / output_chw;
    const int pos_c = pos / output_hw % channel;
    const int pos_h = pos / output_width % output_height;
    const int pos_w = pos % output_width;

    int h_start = pos_h * stride_height - pad_top;
    int w_start = pos_w * stride_width - pad_left;
    const int h_end = MSMIN(h_start + window_height, input_height);
    const int w_end = MSMIN(w_start + window_width, input_width);
    h_start = MSMAX(h_start, 0);
    w_start = MSMAX(w_start, 0);

    int input_start = pos_n * channel * input_height * input_width + pos_c * input_height * input_width;
    int max_idx = h_start * input_width + w_start;
    float max_data = input[input_start + max_idx];

    for (int h_cur = h_start; h_cur < h_end; ++h_cur) {
      for (int w_cur = w_start; w_cur < w_end; ++w_cur) {
        int input_idx = h_cur * input_width + w_cur;
        float input_data = input[input_start + input_idx];
        if (input_data > max_data) {
          max_idx = input_idx;
          max_data = input_data;
        }
      }
    }
    output[pos] = grad[input_start + max_idx];
  }
  return NNACL_OK;
}

int MaxPool3DGradGrad(const float *input, const float *grad, float *output, size_t start, size_t end,
                      Pooling3DParameter *param, PoolingComputeParam *args) {
  PoolingParameter *param_2d = (PoolingParameter *)(param);
  const int channel = args->input_channel_;
  const int input_depth = param->input_d_;
  const int input_height = args->input_h_;
  const int input_width = args->input_w_;

  const int window_depth = param->window_d_;
  const int window_height = args->window_h_;
  const int window_width = args->window_w_;

  const int stride_depth = param->stride_d_;
  const int stride_height = param_2d->stride_h_;
  const int stride_width = param_2d->stride_w_;

  const int pad_front = param->pad_f_;
  const int pad_top = param_2d->pad_u_;
  const int pad_left = param_2d->pad_l_;

  const int output_depth = param->output_d_;
  NNACL_CHECK_ZERO_RETURN_ERR(output_depth);
  const int output_height = args->output_h_;
  NNACL_CHECK_ZERO_RETURN_ERR(output_height);
  const int output_width = args->output_w_;
  NNACL_CHECK_ZERO_RETURN_ERR(output_width);

  const int output_cdhw = channel * output_depth * output_height * output_width;
  NNACL_CHECK_ZERO_RETURN_ERR(output_cdhw);
  const int output_dhw = output_depth * output_height * output_width;
  NNACL_CHECK_ZERO_RETURN_ERR(output_dhw);
  const int output_hw = output_height * output_width;
  NNACL_CHECK_ZERO_RETURN_ERR(output_hw);

  for (size_t pos = start; pos < end; pos++) {
    const int pos_n = pos / output_cdhw;
    const int pos_c = pos / output_dhw % channel;
    const int pos_d = pos / output_hw % output_depth;
    const int pos_h = pos / output_width % output_height;
    const int pos_w = pos % output_width;

    int d_start = pos_d * stride_depth - pad_front;
    int h_start = pos_h * stride_height - pad_top;
    int w_start = pos_w * stride_width - pad_left;
    const int d_end = MSMIN(d_start + window_depth, input_depth);
    const int h_end = MSMIN(h_start + window_height, input_height);
    const int w_end = MSMIN(w_start + window_width, input_width);
    d_start = MSMAX(d_start, 0);
    h_start = MSMAX(h_start, 0);
    w_start = MSMAX(w_start, 0);

    int input_start =
      pos_n * channel * input_depth * input_height * input_width + pos_c * input_depth * input_height * input_width;
    int max_idx = d_start * input_height * input_width + h_start * input_width + w_start;
    float max_data = input[input_start + max_idx];

    for (int d_cur = d_start; d_cur < d_end; ++d_cur) {
      for (int h_cur = h_start; h_cur < h_end; ++h_cur) {
        for (int w_cur = w_start; w_cur < w_end; ++w_cur) {
          int input_idx = d_cur * input_height * input_width + h_cur * input_width + w_cur;
          float input_data = input[input_start + input_idx];
          if (input_data > max_data) {
            max_idx = input_idx;
            max_data = input_data;
          }
        }
      }
    }
    output[pos] = grad[input_start + max_idx];
  }
  return NNACL_OK;
}
