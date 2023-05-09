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

#include "extendrt/session/bisheng_session.h"
#include <vector>
#include <algorithm>
#include <string>
#include <memory>
#include "extendrt/utils/tensor_utils.h"
#include "extendrt/delegate/factory.h"
#include "extendrt/session/factory.h"
#include "extendrt/utils/tensor_default_impl.h"
#include "extendrt/delegate/bisheng/delegate.h"
#include "src/common/log_adapter.h"
#include "src/litert/cxx_api/converters.h"
#include "ir/graph_utils.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
Status BishengSession::Init(const std::shared_ptr<Context> &context, const ConfigInfos &config_info) {
  MS_LOG(INFO) << "BishengSession::Init";
  context_ = ContextUtils::Convert(context.get());
  return kSuccess;
}

Status BishengSession::FindGraphInputs(const std::vector<AnfNodePtr> &node_list,
                                       const std::vector<AnfNodePtr> &graph_inputs,
                                       const std::vector<std::shared_ptr<kernel::BaseKernel>> &kernels) {
  if (graph_inputs.empty()) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::Schedule get graph inputs node failed";
    return kLiteError;
  }

  size_t found_input_node = 0;
  this->inputs_.resize(graph_inputs.size());
  for (size_t ni = 0; ni < node_list.size(); ni++) {
    auto &node = node_list[ni];
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = utils::cast<CNodePtr>(node);
    for (size_t i = 0; i < cnode->inputs().size(); i++) {
      for (size_t j = 0; j < graph_inputs.size(); j++) {
        if (cnode->input(i) == graph_inputs[j]) {
          this->inputs_[j] = kernels[ni]->in_tensors().at(i);
          found_input_node++;
        }
      }
    }
  }
  if (found_input_node != graph_inputs.size()) {
    MS_LOG(ERROR) << "Can not find corresponding anfnode for all funcgraph inputs.";
    return kLiteError;
  }
  return kSuccess;
}

Status BishengSession::FindGraphOutputs(const std::vector<AnfNodePtr> &node_list, const AnfNodePtr &graph_output,
                                        const std::vector<std::shared_ptr<kernel::BaseKernel>> &kernels) {
  if (graph_output == nullptr) {
    MS_LOG(ERROR) << "get graph output node failed.";
    return kLiteError;
  }
  const PrimitiveSet prims{prim::kPrimTupleGetItem, prim::kPrimListGetItem, prim::kPrimArrayGetItem,
                           prim::kPrimMakeTuple};
  auto cnode = utils::cast<CNodePtr>(graph_output);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "bisheng delegate not support empty subgraph now.";
    return kLiteError;
  }
  auto prim_vnode = cnode->input(0);
  if (IsOneOfPrimitive(prim_vnode, prims)) {
    MS_LOG(ERROR) << "bisheng delegate not support maketuple and tuple-get-item operator now.";
    return kLiteError;
  }
  for (size_t ni = 0; ni < node_list.size(); ni++) {
    if (node_list[ni] == graph_output) {
      for (auto &output : kernels_[ni]->out_tensors()) {
        this->outputs_.emplace_back(output);
      }
      break;
    }
  }
  return kSuccess;
}

Status BishengSession::CompileGraph(FuncGraphPtr func_graph, const void *data, size_t size, uint32_t *graph_id) {
  MS_LOG(INFO) << "BishengSession::CompileGraph";
  if (delegate_ == nullptr) {
    MS_LOG(ERROR) << "bisheng delegate not inited";
    return kLiteNullptr;
  }
  // call delegate replace nodes make the delegate replace the graph nodes
  delegate_->ReplaceNodes(func_graph);
  auto nodes = TopoSort(func_graph->get_return());
  // for all the nodes in the graph, call the delegate isDelegateNode and CreateKernel interface to create kernels
  for (auto &node : nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }

    if (!delegate_->IsDelegateNode(node)) {
      MS_LOG(ERROR) << "bisheng session requires all node can be delegated.";
      return kLiteError;
    }
    auto kernel = delegate_->CreateKernel(node);
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "delegate create kernel failed.";
      return kLiteError;
    }
    kernels_.emplace_back(kernel);
  }
  if (kernels_.empty()) {
    MS_LOG(ERROR) << "delegate not support empty subgraph now.";
    return kLiteError;
  }

  auto findio_ret = FindGraphInputs(nodes, func_graph->get_inputs(), kernels_);
  if (findio_ret != kSuccess) {
    MS_LOG(ERROR) << "Search graph input tensors failed.";
    return findio_ret;
  }
  findio_ret = FindGraphOutputs(nodes, func_graph->output(), kernels_);
  if (findio_ret != kSuccess) {
    MS_LOG(ERROR) << "Search graph output tensors failed.";
    return findio_ret;
  }

  // call kernel prepare
  for (auto &kernel : kernels_) {
    auto ret = kernel->Prepare();
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "kernel prepare failed with " << ret;
      return kLiteError;
    }
  }

  return kSuccess;
}

namespace {
void ResetTensorData(const std::vector<void *> &old_data, const std::vector<lite::Tensor *> &tensors) {
  for (size_t j = 0; j < old_data.size(); j++) {
    tensors.at(j)->set_data(old_data.at(j));
  }
}

Status RefDataFromOuter(const std::vector<tensor::Tensor> &outer_tensors,
                        const std::vector<infer::abstract::Tensor *> &inner_tensors) {
  if (outer_tensors.size() != inner_tensors.size()) {
    MS_LOG(EXCEPTION) << "user input size " << outer_tensors.size() << " is not equal to graph input size "
                      << inner_tensors.size();
  }
  std::vector<void *> old_data;
  for (size_t i = 0; i < outer_tensors.size(); i++) {
    auto &user_input = outer_tensors.at(i);
    auto input = inner_tensors.at(i);
    if (user_input.data_type() != input->data_type()) {
      ResetTensorData(old_data, inner_tensors);
      MS_LOG(ERROR) << "Tensor " << user_input.id() << " has a different data type from input" << input->tensor_name()
                    << ".";
      return kLiteError;
    }
    if (user_input.data_c() == nullptr) {
      ResetTensorData(old_data, inner_tensors);
      MS_LOG(ERROR) << "Tensor " << user_input.id() << " has no data.";
      return kLiteError;
    }
    old_data.push_back(input->data());
    if (input->data_type() == kObjectTypeString) {
      MS_LOG(ERROR) << "Not support string type tensor now!";
      return kLiteError;
    }
    if (user_input.data_c() != input->data()) {
      if (input->Size() != user_input.Size()) {
        ResetTensorData(old_data, inner_tensors);
        MS_LOG(ERROR) << "Tensor " << user_input.id() << " has wrong data size.";
        return kLiteError;
      }
      input->set_data(user_input.data_c(), false);
    }
  }
  return kSuccess;
}

std::vector<mindspore::tensor::Tensor> LiteTensorToTensor(const std::vector<infer::abstract::Tensor *> &inner_tensors) {
  std::vector<mindspore::tensor::Tensor> tensors;
  for (auto inner_tensor : inner_tensors) {
    if (inner_tensor == nullptr) {
      MS_LOG(ERROR) << "Input inner_tensors has nullptr.";
      return std::vector<mindspore::tensor::Tensor>{};
    }
    auto type_id = inner_tensor->data_type();
    auto shape = inner_tensor->shape();
    auto data = inner_tensor->MutableData();
    auto data_size = inner_tensor->Size();
    auto ref_tensor_data = std::make_shared<TensorRefData>(data, inner_tensor->ElementsNum(), data_size, shape.size());
    std::vector<int64_t> shape64;
    std::transform(shape.begin(), shape.end(), std::back_inserter(shape64),
                   [](int dim) { return static_cast<int64_t>(dim); });
    mindspore::tensor::Tensor tensor(type_id, shape64, ref_tensor_data);
    auto device_address = inner_tensor->device_data();
    if (device_address != nullptr) {
      auto lite_device_address = std::make_shared<LiteDeviceAddress>(device_address, inner_tensor->Size());
      tensor.set_device_address(lite_device_address);
    }
    tensors.emplace_back(std::move(tensor));
  }
  return tensors;
}
}  // namespace

Status BishengSession::RunGraph(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                                std::vector<tensor::Tensor> *outputs, const MSKernelCallBack &before,
                                const MSKernelCallBack &after) {
  MS_LOG(INFO) << "BishengSession::RunGraph";

  // get inputs and outputs tensors, set the data ptr for inputs
  auto ret = RefDataFromOuter(inputs, this->inputs_);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Sync tensor data from use tensor failed: " << ret;
    return ret;
  }
  // call kernel execute interface one by one
  for (auto &kernel : kernels_) {
    auto exec_ret = kernel->Execute();
    if (exec_ret != kSuccess) {
      MS_LOG(ERROR) << "kernel execute failed with " << exec_ret;
      return kLiteError;
    }
  }

  // copy the data from outputs to user outputs tensors
  *outputs = LiteTensorToTensor(this->outputs_);
  if (outputs->size() != this->outputs_.size()) {
    MS_LOG(ERROR) << "Convert output tensors failed";
    return kLiteNullptr;
  }
  return kSuccess;
}

Status BishengSession::RunGraph(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                                std::vector<tensor::Tensor> *outputs) {
  return RunGraph(graph_id, inputs, outputs, nullptr, nullptr);
}

Status BishengSession::Resize(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                              const std::vector<std::vector<int64_t>> &new_shapes) {
  MS_LOG(EXCEPTION) << "BishengSession::Resize not implemented";
}

std::vector<MutableTensorImplPtr> BishengSession::GetOutputs(uint32_t graph_id) {
  std::vector<MutableTensorImplPtr> result;
  std::transform(this->outputs_.begin(), this->outputs_.end(), std::back_inserter(result),
                 [](infer::abstract::Tensor *output) { return std::make_shared<LiteTensorImpl>(output); });
  return result;
}
std::vector<MutableTensorImplPtr> BishengSession::GetInputs(uint32_t graph_id) {
  std::vector<MutableTensorImplPtr> result;
  std::transform(this->inputs_.begin(), this->inputs_.end(), std::back_inserter(result),
                 [](infer::abstract::Tensor *input) { return std::make_shared<LiteTensorImpl>(input); });
  return result;
}
std::vector<std::string> BishengSession::GetOutputNames(uint32_t graph_id) {
  MS_LOG(EXCEPTION) << "BishengSession::GetOutputNames not implemented";
}
std::vector<std::string> BishengSession::GetInputNames(uint32_t graph_id) {
  MS_LOG(EXCEPTION) << "BishengSession::GetInputNames not implemented";
}
MutableTensorImplPtr BishengSession::GetOutputByTensorName(uint32_t graph_id, const std::string &tensorName) {
  MS_LOG(EXCEPTION) << "BishengSession::GetOutputByTensorName not implemented";
}
MutableTensorImplPtr BishengSession::GetInputByTensorName(uint32_t graph_id, const std::string &name) {
  MS_LOG(EXCEPTION) << "BishengSession::GetInputByTensorName not implemented";
}

static std::shared_ptr<InferSession> BishengSessionCreator(const std::shared_ptr<Context> &ctx,
                                                           const ConfigInfos &config_infos) {
  auto &device_contexts = ctx->MutableDeviceInfo();
  if (device_contexts.empty()) {
    return nullptr;
  }
  auto provider = device_contexts.at(0)->GetProvider();

  auto delegate = std::make_shared<mindspore::BishengDelegate>();
  if (delegate == nullptr) {
    return nullptr;
  }
  auto session = std::make_shared<BishengSession>(delegate);
  constexpr auto kAscendProviderBisheng = "bisheng";
  if (provider != kAscendProviderBisheng) {
    session->Init(ctx);
  }
  return session;
}

REG_SESSION(kBishengSession, BishengSessionCreator);
}  // namespace mindspore
