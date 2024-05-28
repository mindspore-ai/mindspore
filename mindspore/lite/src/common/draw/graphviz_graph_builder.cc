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

#include "src/common/draw/graphviz_graph_builder.h"
#include <set>
#include <vector>
#include "src/common/draw/adapter_graph.h"
#include "ir/dtype.h"

namespace mindspore::lite {
namespace {
inline void StrReplace(std::string *str) {
  replace(str->begin(), str->end(), '/', '_');
  replace(str->begin(), str->end(), '-', '_');
}

inline void ShortName(std::string *str) {
  auto pos = str->rfind('/');
  if (pos == std::string::npos) {
    return;
  }
  *str = str->substr(pos + 1);
}

inline std::string GetNodeId(const AdapterNode &node) {
  auto name = node.GetName();
  StrReplace(&name);
  return name;
}

inline std::string GetNodeLabel(const AdapterNode &node) {
  auto name = node.GetName();
  ShortName(&name);
  StrReplace(&name);
  return name;
}

inline std::string GetTensorId(const lite::Tensor &tensor) {
  auto name = tensor.tensor_name();
  StrReplace(&name);
  return name;
}

inline std::string GetTensorInfo(const lite::Tensor &tensor) {
  auto tensor_info = FormatEnumToString(tensor.format());
  tensor_info += ", ";
  tensor_info += TypeIdToString(tensor.data_type());
  tensor_info += ", ";
  tensor_info += lite::ShapeVectorToStr(tensor.shape());
  return tensor_info;
}
}  // namespace

std::shared_ptr<GVGraph> GVGraphBuilder::Build(const std::shared_ptr<AdapterGraph> &graph) {
  gv_graph_ = std::make_shared<GVGraph>(graph->GetName());
  // graph inputs
  for (auto in_tensor : graph->GetInputs()) {
    this->AppendGraphInputNode(*in_tensor);
  }
  // nodes
  for (const auto *node : graph->GetNodes()) {
    auto node_id = GetNodeId(*node);
    auto node_label = GetNodeLabel(*node);
    for (size_t i = 0; i < node->InputSize(); i++) {
      auto in_tensor = node->GetInput(i);
      if (GetBelongingGVNode(in_tensor).first == nullptr) {
        if (!in_tensor->IsConst()) {
          MS_LOG(WARNING) << "The " << i << "th input of " << node->GetName()
                          << " is neither a const tensor nor an output of other node. Treat it as a weight node.";
        }
        auto tensor_id = node_id + "_in_" + std::to_string(i);
        auto tensor_label = node_label + "_in_" + std::to_string(i);
        AppendWeightNode(*in_tensor, tensor_id, tensor_label);
      }
    }
    auto ret = this->AppendComputeNode(*node);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Create and append gv_node for " << node->GetName() << " failed.";
      return nullptr;
    }
  }
  // graph outputs
  auto ret = this->AppendGraphOutputNode(graph->GetOutputs());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Create and append graph return node failed";
    return nullptr;
  }
  return this->gv_graph_;
}

void GVGraphBuilder::AppendGraphInputNode(const lite::Tensor &tensor) {
  auto tensor_id = GetTensorId(tensor);
  auto gv_node = lite::GVNode::CreateInput(tensor_id, {tensor_id}, {GetTensorInfo(tensor)});
  MS_ASSERT(gv_node != nullptr);
  gv_graph_->AppendNode(gv_node);
  gv_node_out_tensor_map_[&tensor] = std::make_pair(gv_node, 0);
}

namespace {
template <typename T>
std::string BufferToString(const T *buffer, size_t size) {
  MS_ASSERT(buffer != nullptr);
  constexpr size_t print_pre_number = 3;
  constexpr size_t print_post_number = 3;
  constexpr size_t print_period_number = 2;
  if (size <= print_pre_number + print_post_number + print_period_number) {
    std::ostringstream oss;
    for (size_t i = 0; i < size; i++) {
      if (i == 0) {
        oss << buffer[i];
      } else {
        oss << ", " << buffer[i];
      }
    }
    return oss.str();
  }

  size_t index = 0;
  std::ostringstream oss;
  for (size_t i = 0; i < print_pre_number; i++, index++) {
    if (index == 0) {
      oss << buffer[index];
    } else {
      oss << ", " << buffer[index];
    }
  }
  oss << "...";
  for (size_t i = 0; i < print_post_number; i++, index++) {
    oss << ", " << buffer[index];
  }
  return oss.str();
}

std::string TensorDataString(const lite::Tensor &tensor) {
  if (tensor.shape().size() != 1 || tensor.shape()[0] <= 0 || tensor.data() == nullptr) {
    return "";
  }
  auto data_size = static_cast<size_t>(tensor.shape()[0]);

  std::ostringstream oss;
  oss << "\n[";
  if (tensor.data_type() == kNumberTypeInt || tensor.data_type() == kNumberTypeInt32) {
    auto data = reinterpret_cast<int *>(tensor.data());
    oss << BufferToString(data, data_size);
  } else if (tensor.data_type() == kNumberTypeInt64) {
    auto data = reinterpret_cast<int64_t *>(tensor.data());
    oss << BufferToString(data, data_size);
  } else {
    return "";
  }
  oss << "]";
  return oss.str();
}
}  // namespace

void GVGraphBuilder::AppendWeightNode(const lite::Tensor &tensor, const std::string &id, const std::string &label) {
  auto gv_node = lite::GVNode::CreateWeight(id, label + TensorDataString(tensor), {id}, {GetTensorInfo(tensor)});
  MS_ASSERT(gv_node != nullptr);
  gv_graph_->AppendNode(gv_node);
  AppendOutTensorMap(&tensor, gv_node, 0);
}

int GVGraphBuilder::AppendComputeNode(const AdapterNode &node) {
  auto gv_node = CreateComputeNode(node);
  if (gv_node == nullptr) {
    MS_LOG(ERROR) << "Create gv_node for " << node.GetName() << " failed.";
    return RET_ERROR;
  }
  gv_graph_->AppendNode(gv_node);
  for (size_t i = 0; i < node.OutputSize(); i++) {
    AppendOutTensorMap(node.GetOutput(i), gv_node, i);
  }
  auto ret = LinkNodes(node, *gv_node);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Link inputs for " << node.GetName() << " failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int GVGraphBuilder::AppendGraphOutputNode(const std::vector<lite::Tensor *> &out_tensors) {
  auto out_tensor_size = out_tensors.size();
  auto gv_node = lite::GVNode::CreateOutput("return", out_tensor_size);
  if (gv_node == nullptr) {
    MS_LOG(ERROR) << "create output node failed!";
    return RET_ERROR;
  }
  gv_graph_->AppendNode(gv_node);
  for (size_t i = 0; i < out_tensors.size(); i++) {
    auto out_tensor = out_tensors[i];
    auto pair = this->GetBelongingGVNode(out_tensor);
    if (pair.first == nullptr) {
      MS_LOG(ERROR) << "Can not find graph output tensor source: " << out_tensor->tensor_name();
      return RET_ERROR;
    }
    auto link_ret = gv_graph_->Link(pair.first->name(), pair.second, gv_node->name(), i);
    if (link_ret != RET_OK) {
      MS_LOG(ERROR) << "Link " << i << "th input tensor of return failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

GVNode *GVGraphBuilder::CreateComputeNode(const AdapterNode &node) {
  auto node_id = GetNodeId(node);
  auto node_label = GetNodeLabel(node);
  std::vector<std::string> output_names;
  std::vector<std::string> output_infos;
  for (auto out_tensor : node.GetOutputs()) {
    output_names.emplace_back(GetTensorId(*out_tensor));
    output_infos.emplace_back(GetTensorInfo(*out_tensor));
  }
  auto *gv_node =
    lite::GVNode::CreateCNode(node_id, node_label, node.InputSize(), output_names, output_infos, node.IsHighlight());
  MS_ASSERT(gv_node != nullptr);
  return gv_node;
}

void GVGraphBuilder::AppendOutTensorMap(const lite::Tensor *tensor, lite::GVNode *node, size_t out_index) {
  gv_node_out_tensor_map_[tensor] = std::make_pair(node, out_index);
}

std::pair<lite::GVNode *, size_t> GVGraphBuilder::GetBelongingGVNode(const lite::Tensor *tensor) const {
  auto iter = gv_node_out_tensor_map_.find(tensor);
  if (iter == gv_node_out_tensor_map_.end()) {
    return {};
  } else {
    return iter->second;
  }
}
int GVGraphBuilder::LinkNodes(const AdapterNode &node, const GVNode &gv_node) {
  for (size_t i = 0; i < node.InputSize(); i++) {
    auto in_tensor = node.GetInput(i);
    auto pair = this->GetBelongingGVNode(in_tensor);
    if (pair.first == nullptr) {
      MS_LOG(ERROR) << "Can not find input tensor source: " << in_tensor->tensor_name();
      return RET_ERROR;
    }
    auto link_ret = gv_graph_->Link(pair.first->name(), pair.second, gv_node.name(), i);
    if (link_ret != RET_OK) {
      MS_LOG(ERROR) << "Link " << i << "th input tensor of " << node.GetName() << " failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
