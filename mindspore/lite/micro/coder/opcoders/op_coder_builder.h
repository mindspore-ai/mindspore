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
#ifndef MINDSPORE_LITE_MICRO_OPCODERS_OP_CODER_BUILDER_H_
#define MINDSPORE_LITE_MICRO_OPCODERS_OP_CODER_BUILDER_H_

#include <vector>
#include <memory>
#include "coder/opcoders/op_coder.h"
#include "micro/coder/allocator/allocator.h"

namespace mindspore::lite::micro {

class OpCoderBuilder {
 public:
  std::unique_ptr<OperatorCoder> build();

  OpCoderBuilder &inputs(const std::vector<Tensor *> &inputs);

  OpCoderBuilder &outputs(const std::vector<Tensor *> &outputs);

  OpCoderBuilder &node(const Model::Node *node);

  OpCoderBuilder &data_type(TypeId data_type);

  OpCoderBuilder &mode(CodeMode mode);

  OpCoderBuilder &input_indices(const std::vector<uint32_t> &indices);

  OpCoderBuilder &output_indices(const std::vector<uint32_t> &indices);

  OpCoderBuilder &target(Target target);

  OpCoderBuilder &support_parallel(bool parallel);

  void Reset();

 private:
  std::vector<Tensor *> inputs_;

  std::vector<Tensor *> outputs_;

  const mindspore::lite::Model::Node *node_ = nullptr;

  size_t node_index_{0};

  Target target_{kTargetUnknown};

  TypeId data_type_{kTypeUnknown};

  CodeMode mode_{Code_Unknown};

  std::vector<uint32_t> input_indices_;

  std::vector<uint32_t> output_indices_;

  bool support_parallel_{false};
};

}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_MICRO_OPCODERS_OP_CODER_BUILDER_H_
