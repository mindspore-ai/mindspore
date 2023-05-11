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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_CONVOLUTION_WINOGRAD_FP16_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_CONVOLUTION_WINOGRAD_FP16_CODER_H_

#include <memory>
#include <string>
#include <vector>
#include "coder/opcoders/nnacl/fp32/convolution_winograd_fp32_coder.h"
#include "nnacl/conv_parameter.h"

namespace mindspore::lite::micro::nnacl {
typedef struct TransFuncFp16Str {
  std::string in_func_;
  std::string in_step_func_;
  std::string in_pack_func_;
  std::string out_func_;
} TransFuncFp16Str;

class ConvolutionWinogradFP16Coder : public ConvolutionWinogradFP32Coder {
 public:
  ConvolutionWinogradFP16Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                               const LiteGraph::Node *node, size_t node_index, Target target, int output_unit)
      : ConvolutionWinogradFP32Coder(in_tensors, out_tensors, node, node_index, target, output_unit) {
    is_weight_online_ = true;
    data_type_ = kNumberTypeFloat16;
  }

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

  ~ConvolutionWinogradFP16Coder() override = default;

 private:
  void InitCodeOnline(CoderContext *const context) override;
  int ConfigInputOutput() override;
  std::string GetInputTransFunc(int input_unit) override;
  std::string GetInputTransStepFunc(int input_unit);
  std::string GetInputTransPackFunc(int input_unit);
  std::string GetOutputTransFunc(int input_unit, int output_unit, ActType act_type) override;
  void CollectFilesForFunc(CoderContext *const context) override;
  int InitTmpBuffer() override;
  TransFuncFp16Str trans_func_str_;
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_CONVOLUTION_WINOGRAD_FP16_CODER_H_
