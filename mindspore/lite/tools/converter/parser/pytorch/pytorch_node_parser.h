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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_PYTORCH_PYTORCH_NODE_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_PYTORCH_PYTORCH_NODE_PARSER_H_

#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include "torch/script.h"
#include "include/errorcode.h"
#include "ir/dtype/type_id.h"
#include "ops/primitive_c.h"
#include "ops/op_name.h"
#include "src/common/log_adapter.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/parser/parser_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
class PytorchNodeParser {
 public:
  explicit PytorchNodeParser(std::string node_name) : name_(std::move(node_name)) {}

  virtual ~PytorchNodeParser() = default;

  virtual PrimitiveCPtr Parse(const torch::jit::Node *torch_node, std::vector<size_t> *input_indices) {
    return nullptr;
  }

  static std::string GetTorchNodeType(const torch::jit::Node *torch_node);

  static TypeId GetDataTypeFromTorch(const at::ScalarType torch_data_type);

  template <typename T>
  static T GetValueFromConstNode(const torch::jit::Value *value_node) {
    T data{};
    auto ivalue = torch::jit::toIValue(value_node);
    MS_CHECK_TRUE_RET(ivalue.has_value(), data);
    auto value = ivalue.value();
    auto optional_value = value.toOptional<T>();
    MS_CHECK_TRUE_RET(optional_value, data);
    return optional_value.value();
  }

 protected:
  const std::string name_{};
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_PYTORCH_PYTORCH_NODE_PARSER_H_
