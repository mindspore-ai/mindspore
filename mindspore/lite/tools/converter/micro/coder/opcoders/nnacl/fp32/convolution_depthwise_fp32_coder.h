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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP32_CONVOLUTION_DEPTHWISE_FP32_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP32_CONVOLUTION_DEPTHWISE_FP32_CODER_H_

#include <vector>
#include "coder/opcoders/base/conv2d_base_coder.h"
#include "src/litert/kernel/cpu/fp32/convolution_depthwise_fp32.h"

namespace mindspore::lite::micro::nnacl {
class ConvolutionDepthwiseFP32Coder : public Conv2DBaseCoder {
 public:
  ConvolutionDepthwiseFP32Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                                const LiteGraph::Node *node, size_t node_index, Target target)
      : Conv2DBaseCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~ConvolutionDepthwiseFP32Coder() override = default;
  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

 private:
  int InitWeightBiasOffline();
  int InitWeightBiasOnline();
  int InitParameter();

 protected:
  virtual void InitCodeOnline(CoderContext *const context);
  virtual void CollectFilesForFunc(CoderContext *const context);
  size_t packed_weight_size_{0};
  void *packed_weight_{nullptr};
  size_t packed_bias_size_{0};
  void *bias_{nullptr};
  bool is_weight_online_{false};
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP32_CONVOLUTION_DEPTHWISE_FP32_CODER_H_
