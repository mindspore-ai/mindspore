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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_NNACL_FP32_CONVOLUTION_FP32_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_NNACL_FP32_CONVOLUTION_FP32_CODER_H_

#include <vector>
#include <string>
#include "nnacl/conv_parameter.h"
#include "coder/opcoders/base/conv2d_base_coder.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"

namespace mindspore::lite::micro::nnacl {
class ConvolutionFP32Coder final : public Conv2DBaseCoder {
 public:
  ConvolutionFP32Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                       const Model::Node *node, size_t node_index, Target target)
      : Conv2DBaseCoder(in_tensors, out_tensors, node, node_index, target) {}

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

  ~ConvolutionFP32Coder() override = default;

 private:
  int InitWeightBias(CoderContext *const context);

  int InitTmpBuffer();

  int Resize();

  float *packed_weight_{nullptr};

  float *bias_data_{nullptr};

  float *packed_input_{nullptr};

  size_t packed_input_size_{0};

  bool de_quant_flag_{false};

  int thread_count_{0};

  float *col_major_input_{nullptr};
  size_t col_major_input_size_{0};

  size_t pack_weight_size_{0};
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_NNACL_FP32_CONVOLUTION_FP32_CODER_H_
