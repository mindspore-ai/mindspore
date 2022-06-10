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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "nnacl/fp32/maxpool_with_argmax.h"

int MaxPoolWithArgmax(const float *input, float *output, int *index, size_t start, size_t end,
                      PoolingParameter *param) {
  const int channel = param->input_channel_;
  const int input_height = param->input_h_;
  const int input_width = param->input_w_;
  const int window_height = param->window_h_;
  const int window_width = param->window_w_;
  const int stride_height = param->stride_h_;
  const int stride_width = param->stride_w_;
  const int pad_top = param->pad_u_;
  const int pad_left = param->pad_l_;
  const int output_height = param->output_h_;
  NNACL_CHECK_ZERO_RETURN_ERR(output_height);
  const int output_width = param->output_w_;
  NNACL_CHECK_ZERO_RETURN_ERR(output_width);

  const int output_chw = channel * output_height * output_width;
  NNACL_CHECK_ZERO_RETURN_ERR(output_chw);
  const int output_hw = output_height * output_width;
  NNACL_CHECK_ZERO_RETURN_ERR(output_hw);

  for (size_t pos = start; pos < end; pos++) {
    const int posn = pos / output_chw;
    const int posc = pos / output_hw % channel;
    const int posh = pos / output_height % output_width;
    const int posw = pos % output_width;
    int hstart = posh * stride_height - pad_top;
    int wstart = posw * stride_width - pad_left;
    const int hend = MSMIN(hstart + window_height, input_height);
    const int wend = MSMIN(wstart + window_width, input_width);
    hstart = MSMAX(hstart, 0);
    wstart = MSMAX(wstart, 0);
    int inputStart = posn * channel * input_height * input_width;
    int maxIdx = posc * input_height * input_width + hstart * input_width + wstart;
    float maxData = input[inputStart + maxIdx];
    for (int hcur = hstart; hcur < hend; ++hcur) {
      for (int wcur = wstart; wcur < wend; ++wcur) {
        int inputIdx = posc * input_height * input_width + hcur * input_width + wcur;
        float inputData = input[inputStart + inputIdx];
        if (inputData > maxData) {
          maxIdx = inputIdx;
          maxData = inputData;
        }
      }
    }
    output[pos] = maxData;
    index[pos] = maxIdx;
  }
  return NNACL_OK;
}
