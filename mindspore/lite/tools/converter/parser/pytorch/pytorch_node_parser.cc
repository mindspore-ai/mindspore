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

#include "tools/converter/parser/pytorch/pytorch_node_parser.h"
#include <unordered_map>

namespace mindspore {
namespace lite {
namespace {
static std::unordered_map<at::ScalarType, TypeId> kTorchDataTypeTransferMap = {
  {at::ScalarType::Bool, kNumberTypeBool},     {at::ScalarType::Byte, kNumberTypeUInt8},
  {at::ScalarType::Char, kNumberTypeInt8},     {at::ScalarType::Int, kNumberTypeInt},
  {at::ScalarType::Long, kNumberTypeInt},      {at::ScalarType::Half, kNumberTypeFloat16},
  {at::ScalarType::Float, kNumberTypeFloat32}, {at::ScalarType::Double, kNumberTypeFloat32}};
}  // namespace

std::string PytorchNodeParser::GetTorchNodeType(const torch::jit::Node *torch_node) {
  const auto &kind = torch_node->kind();
  std::string node_type = kind.toUnqualString();
  if (node_type.empty()) {
    return node_type;
  }
  node_type = node_type.at(0) == '_' ? node_type.substr(1) : node_type;
  node_type = node_type.at(node_type.size() - 1) == '_' ? node_type.substr(0, node_type.size() - 1) : node_type;
  return node_type;
}

TypeId PytorchNodeParser::GetDataTypeFromTorch(const at::ScalarType torch_data_type) {
  auto iter = kTorchDataTypeTransferMap.find(torch_data_type);
  if (iter == kTorchDataTypeTransferMap.end()) {
    MS_LOG(ERROR) << "Unsupported torch data type: " << torch_data_type;
    return kTypeUnknown;
  }
  return iter->second;
}
}  // namespace lite
}  // namespace mindspore
