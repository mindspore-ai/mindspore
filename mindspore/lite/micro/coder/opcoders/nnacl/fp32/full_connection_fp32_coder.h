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
#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_NNACL_FP32_FULL_CONNECTION_FP32_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_NNACL_FP32_FULL_CONNECTION_FP32_CODER_H_

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "coder/opcoders/nnacl/fp32/matmul_fp32_base_coder.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"

namespace mindspore::lite::micro::nnacl {

class FullConnectionFP32Coder final : public MatMulFP32BaseCoder {
 public:
  FullConnectionFP32Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                          const Model::Node *node, size_t node_index, Target target)
      : MatMulFP32BaseCoder(in_tensors, out_tensors, node, node_index, target) {}

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

  ~FullConnectionFP32Coder() override = default;

 private:
  int Init() override;
  int ReSize() override;
};

}  // namespace mindspore::lite::micro::nnacl

#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_NNACL_FP32_FULL_CONNECTION_FP32_CODER_H_
