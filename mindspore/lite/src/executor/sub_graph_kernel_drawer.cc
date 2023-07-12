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

#include "src/executor/sub_graph_kernel_drawer.h"
#include <set>
#include <vector>

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

inline std::string NodeNameForDraw(const kernel::KernelExec &kernel) {
  auto name = kernel.name();
  ShortName(&name);
  StrReplace(&name);
  return name;
}

inline std::string TensorNameForDraw(const lite::Tensor &tensor) {
  auto name = tensor.tensor_name();
  StrReplace(&name);
  return name;
}

inline std::string TensorInfoForDraw(const lite::Tensor &tensor) {
  auto tensor_info = FormatEnumToString(tensor.format());
  tensor_info += ", ";
  tensor_info += lite::ShapeVectorToStr(tensor.shape());
  return tensor_info;
}
}  // namespace

std::shared_ptr<SubGraphKernelGVGraph> SubGraphKernelGVGraph::Create(
  const kernel::SubGraphKernel &sub_graph, const std::vector<schema::PrimitiveType> &mark_types) {
  return SubGraphKernelGVGraph::Create(
    sub_graph, [&mark_types](const kernel::KernelExec &kernel) { return IsContain(mark_types, kernel.type()); });
}

std::shared_ptr<SubGraphKernelGVGraph> SubGraphKernelGVGraph::Create(const kernel::SubGraphKernel &sub_graph,
                                                                     const MarkFilter &mark_filter) {
  auto graph = std::make_shared<SubGraphKernelGVGraph>(sub_graph.name());
  // graph inputs
  for (auto in_tensor : sub_graph.in_tensors()) {
    graph->AppendGraphInputNode(*in_tensor);
  }
  // nodes
  for (const auto *node : sub_graph.immutable_nodes()) {
    auto node_name = NodeNameForDraw(*node);
    for (size_t i = 0; i < node->in_tensors().size(); i++) {
      auto in_tensor = node->in_tensors()[i];
      if (in_tensor->IsConst()) {
        auto tensor_name = node_name + "_in_" + std::to_string(i);
        auto ret = graph->AppendWeightNode(*in_tensor, tensor_name);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "Create and append " << i << "th  weight node of " << node->name() << " failed.";
          return nullptr;
        }
      }
    }
    auto ret = graph->AppendKernelExecNode(*node, mark_filter != nullptr && mark_filter(*node));
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Create and append gv_node for " << node->name() << " failed.";
      return nullptr;
    }
  }
  // graph outputs
  auto ret = graph->AppendGraphOutputNode(sub_graph.out_tensors());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Create and append graph return node failed";
    return nullptr;
  }
  return graph;
}

void SubGraphKernelGVGraph::AppendGraphInputNode(const lite::Tensor &tensor) {
  auto tensor_name = TensorNameForDraw(tensor);
  auto gv_node = lite::GVNode::CreateInput(tensor_name, {tensor_name}, {TensorInfoForDraw(tensor)});
  MS_ASSERT(gv_node != nullptr);
  AppendNode(gv_node);
  gv_node_out_tensor_map_[&tensor] = std::make_pair(gv_node, 0);
}

int SubGraphKernelGVGraph::AppendWeightNode(const lite::Tensor &tensor, const std::string &name) {
  if (MS_UNLIKELY(!tensor.IsConst())) {
    MS_LOG(ERROR) << "Input `tensor` is not a const tensor.";
    return RET_ERROR;
  }
  auto gv_node = lite::GVNode::CreateWeight(name, {name}, {TensorInfoForDraw(tensor)});
  MS_ASSERT(gv_node != nullptr);
  AppendNode(gv_node);
  AppendOutTensorMap(&tensor, gv_node, 0);
  return RET_OK;
}

int SubGraphKernelGVGraph::AppendKernelExecNode(const kernel::KernelExec &kernel, bool highlight) {
  auto gv_node = CreateKernelExecNode(kernel, highlight);
  if (gv_node == nullptr) {
    MS_LOG(ERROR) << "Create gv_node for " << kernel.name() << " failed.";
    return RET_ERROR;
  }
  AppendNode(gv_node);
  for (size_t i = 0; i < kernel.out_tensors().size(); i++) {
    AppendOutTensorMap(kernel.out_tensors()[i], gv_node, i);
  }
  auto ret = LinkNodes(kernel, *gv_node);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Link inputs for " << kernel.name() << " failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int SubGraphKernelGVGraph::AppendGraphOutputNode(const std::vector<lite::Tensor *> &out_tensors) {
  auto out_tensor_size = out_tensors.size();
  auto gv_node = lite::GVNode::CreateOutput("return", out_tensor_size);
  MS_ASSERT(gv_node != nullptr);
  AppendNode(gv_node);
  for (size_t i = 0; i < out_tensors.size(); i++) {
    auto out_tensor = out_tensors[i];
    auto pair = this->GetBelongingGVNode(out_tensor);
    if (pair.first == nullptr) {
      MS_LOG(ERROR) << "Can not find graph output tensor source: " << out_tensor->tensor_name();
      return RET_ERROR;
    }
    auto link_ret = this->Link(pair.first->name(), pair.second, gv_node->name(), i);
    if (link_ret != RET_OK) {
      MS_LOG(ERROR) << "Link " << i << "th input tensor of return failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

GVNode *SubGraphKernelGVGraph::CreateKernelExecNode(const kernel::KernelExec &kernel, bool highlight) {
  auto node_name = NodeNameForDraw(kernel);
  std::vector<std::string> output_names;
  std::vector<std::string> output_infos;
  for (auto out_tensor : kernel.out_tensors()) {
    output_names.emplace_back(TensorNameForDraw(*out_tensor));
    output_infos.emplace_back(TensorInfoForDraw(*out_tensor));
  }
  auto *gv_node =
    lite::GVNode::CreateCNode(node_name, kernel.in_tensors().size(), output_names, output_infos, highlight);
  MS_ASSERT(gv_node != nullptr);
  return gv_node;
}

void SubGraphKernelGVGraph::AppendOutTensorMap(const lite::Tensor *tensor, lite::GVNode *node, size_t out_index) {
  gv_node_out_tensor_map_[tensor] = std::make_pair(node, out_index);
}

std::pair<lite::GVNode *, size_t> SubGraphKernelGVGraph::GetBelongingGVNode(const lite::Tensor *tensor) const {
  auto iter = gv_node_out_tensor_map_.find(tensor);
  if (iter == gv_node_out_tensor_map_.end()) {
    return {};
  } else {
    return iter->second;
  }
}
int SubGraphKernelGVGraph::LinkNodes(const kernel::KernelExec &kernel, const GVNode &gv_node) {
  for (size_t i = 0; i < kernel.in_tensors().size(); i++) {
    auto in_tensor = kernel.in_tensors()[i];
    auto pair = this->GetBelongingGVNode(in_tensor);
    if (pair.first == nullptr) {
      MS_LOG(ERROR) << "Can not find input tensor source: " << in_tensor->tensor_name();
      return RET_ERROR;
    }
    auto link_ret = this->Link(pair.first->name(), pair.second, gv_node.name(), i);
    if (link_ret != RET_OK) {
      MS_LOG(ERROR) << "Link " << i << "th input tensor of " << kernel.name() << " failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
