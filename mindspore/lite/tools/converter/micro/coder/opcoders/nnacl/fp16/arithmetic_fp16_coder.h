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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_ARITHMETIC_FP16_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_ARITHMETIC_FP16_CODER_H_

#include <vector>
#include <string>
#include "coder/opcoders/nnacl/fp32/arithmetic_fp32_coder.h"
#include "nnacl/base/cast_base.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
namespace mindspore::lite::micro::nnacl {
class ArithmeticFP16Coder final : public ArithmeticFP32Coder {
 public:
  ArithmeticFP16Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                      const LiteGraph::Node *node, size_t node_index, Target target)
      : ArithmeticFP32Coder(in_tensors, out_tensors, node, node_index, target) {}

  ~ArithmeticFP16Coder() override = default;

  int DoCode(CoderContext *const context) override;

 private:
  int Prepare(CoderContext *const context) override;

  int ReSize(CoderContext *const context) override;

  void InitFunTable() override;

  int ExecuteCode(const std::string &input0, const std::string &input1, const std::string &output, int size,
                  CoderContext *const context, NNaclFp32Serializer *const code);
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_ARITHMETIC_FP16_CODER_H_
