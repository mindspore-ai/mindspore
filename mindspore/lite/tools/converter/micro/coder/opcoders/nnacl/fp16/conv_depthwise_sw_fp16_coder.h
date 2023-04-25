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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_CONV_DEPTHWISE_SW_FP16_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_CONV_DEPTHWISE_SW_FP16_CODER_H_

#include <vector>
#include <string>
#include "coder/opcoders/base/conv2d_base_coder.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "base/float16.h"

namespace mindspore::lite::micro::nnacl {
class ConvolutionDepthwiseSWFP16Coder final : public Conv2DBaseCoder {
 public:
  ConvolutionDepthwiseSWFP16Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                                  const LiteGraph::Node *node, size_t node_index, Target target)
      : Conv2DBaseCoder(in_tensors, out_tensors, node, node_index, target) {
    data_type_ = kNumberTypeFloat16;
  }

  ~ConvolutionDepthwiseSWFP16Coder() override = default;

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

 private:
  void CollectFilesForFunc(CoderContext *const context);
  int InitWeightBias(CoderContext *const context);
  SlidingWindowParam *sw_param_{nullptr};
  void *packed_weight_{nullptr};
  void *bias_data_{nullptr};
  float16 *packed_input_{nullptr};
  float16 *packed_output_{nullptr};
  size_t pack_output_size_{0};
  size_t pack_input_size_{0};
  std::string input_ptr_;
  std::string output_ptr_;
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_CONV_DEPTHWISE_SW_FP16_CODER_H_
