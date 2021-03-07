/**
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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_FP32_CONVOLUTION_WINOGRAD_FP32_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_FP32_CONVOLUTION_WINOGRAD_FP32_CODER_H_

#include <memory>
#include <string>
#include <vector>
#include "coder/opcoders/base/conv2d_base_coder.h"
#include "nnacl/conv_parameter.h"

namespace mindspore::lite::micro::nnacl {
class ConvolutionWinogradFP32Coder : public Conv2DBaseCoder {
 public:
  ConvolutionWinogradFP32Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                               const Model::Node *node, size_t node_index, Target target, int output_unit)
      : Conv2DBaseCoder(in_tensors, out_tensors, node, node_index, target), output_unit_(output_unit) {}

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

  ~ConvolutionWinogradFP32Coder() override = default;

 private:
  int InitWeightBias();

  int ConfigInputOutput();

  int InitTmpBuffer();

  int ReSize();

  int WinogradFilterTransform(const float *weight_data, float *matrix_g, const float *matrix_gt, int oc_block);

  std::string GetInputTransFunc(int input_unit);

  std::string GetOutputTransFunc(int input_unit, int output_unit, ActType act_type);

  float *trans_weight_{nullptr};
  float *new_bias_{nullptr};

  int kernel_unit_{0};
  int input_unit_{0};
  int output_unit_{0};

  size_t tmp_data_size_{0};
  size_t tile_buffer_size_{0};
  size_t gemm_out_size_{0};
  size_t col_buffer_size_{0};

  float *tmp_data_{nullptr};
  float *trans_input_{nullptr};
  float *gemm_out_{nullptr};
  float *col_buffer_{nullptr};

  std::string in_func_;
  std::string out_func_;
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_FP32_CONVOLUTION_WINOGRAD_FP32_CODER_H_
