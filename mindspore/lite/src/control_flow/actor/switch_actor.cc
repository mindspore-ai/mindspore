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

#include "src/control_flow/actor/switch_actor.h"
#include <utility>
#include <algorithm>
#include "mindrt/include/mindrt.hpp"
#include "src/runtime/kernel_exec_util.h"
#include "src/common/tensor_util.h"
#include "src/runtime/inner_allocator.h"
#ifdef ENABLE_FP16
#include "src/runtime/kernel/cpu/fp16/fp16_op_handler.h"
#endif
namespace {
const constexpr int kSwitchMaxInputKernelSize = 3;
const constexpr int kSwitchMinInputKernelSize = 2;
const constexpr int kSwitchTruePartialInputIndex = 1;
const constexpr int kSwitchFalsePartialInputIndex = 2;
const constexpr int kSwitchCondTensorIndex = 0;
}  // namespace

namespace mindspore::lite {
int LiteSwitchOpActor::SetSwitchPartialNodes() {
  auto switch_op_input_kernel_size = switch_type_node_->in_kernels().size();
  // special case, switch cond input is const, should be removed in the future.
  if (switch_op_input_kernel_size == kSwitchMinInputKernelSize) {
    // reverse switch node input, then false cast to 0, true cast to 1, which is same as switch layer index.
    partial_nodes_.push_back(switch_type_node_->in_kernels().at(kSwitchFalsePartialInputIndex - 1));
    partial_nodes_.push_back(switch_type_node_->in_kernels().at(kSwitchTruePartialInputIndex - 1));
    return RET_OK;
  }

  if (switch_op_input_kernel_size == kSwitchMaxInputKernelSize) {
    // reverse switch node input.
    partial_nodes_.push_back(switch_type_node_->in_kernels().at(kSwitchFalsePartialInputIndex));
    partial_nodes_.push_back(switch_type_node_->in_kernels().at(kSwitchTruePartialInputIndex));
    return RET_OK;
  }
  MS_LOG(ERROR) << "switch op input kernel size: " << switch_op_input_kernel_size << ", which is not support.";
  return RET_ERROR;
}

int LiteSwitchOpActor::SetSwitchLayerPartialNodes() {
  for (size_t i = 1; i < switch_type_node_->in_kernels().size(); ++i) {
    partial_nodes_.push_back(switch_type_node_->in_kernels()[i]);
  }
  return RET_OK;
}

int LiteSwitchOpActor::GetSwitchAndCallNode(kernel::SubGraphKernel *subgraph_kernel) {
  for (auto &node : subgraph_kernel->nodes()) {
    if (node->type() != schema::PrimitiveType_Call) {
      continue;
    }
    call_node_ = node;
    auto switch_node = kernel::KernelExecUtil::GetInputsSpecificNode(node, schema::PrimitiveType_Switch);
    auto switch_layer_node = kernel::KernelExecUtil::GetInputsSpecificNode(node, schema::PrimitiveType_SwitchLayer);
    if (switch_node != nullptr) {
      switch_type_node_ = switch_node;
      return SetSwitchPartialNodes();
    }
    if (switch_layer_node != nullptr) {
      switch_type_node_ = switch_layer_node;
      return SetSwitchLayerPartialNodes();
    }
  }
  return RET_OK;
}

void LiteSwitchOpActor::AppendOutputTensors() {
  auto output_tensors = kernel_->out_tensors();
  for (auto &partial_node : partial_nodes_) {
    for (auto &tensor : partial_node->in_tensors()) {
      if (std::find(output_tensors.begin(), output_tensors.end(), tensor) == output_tensors.end()) {
        output_tensors.push_back(tensor);
      }
    }
  }
  kernel_->set_out_tensors(output_tensors);
}

int LiteSwitchOpActor::ModifySubgraphKernel() {
  auto *subgraph_kernel = reinterpret_cast<kernel::SubGraphKernel *>(kernel_);
  if (subgraph_kernel == nullptr) {
    MS_LOG(INFO) << "kernel is not subgraph kernel, no partial call.";
    return RET_OK;
  }

  int ret = GetSwitchAndCallNode(subgraph_kernel);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GetSwitchAndCallCnode failed.";
    return ret;
  }

  subgraph_kernel->DropNode(call_node_);
  subgraph_kernel->DropNode(switch_type_node_);
  for (auto &partial_node : partial_nodes_) {
    subgraph_kernel->DropNode(partial_node);
  }
  return ret;
}

int LiteSwitchOpActor::UpdateActorOutput() {
  if (call_node_ == nullptr) {
    MS_LOG(ERROR) << "not get the call node.";
    return RET_ERROR;
  }
  auto call_output_tensors = call_node_->out_tensors();
  auto output_tensors = kernel_->out_tensors();
  for (auto iter = output_tensors.begin(); iter != output_tensors.end();) {
    if (IsContain(call_output_tensors, *iter)) {
      iter = output_tensors.erase(iter);
    } else {
      ++iter;
    }
  }
  kernel_->set_out_tensors(output_tensors);
  return RET_OK;
}

int LiteSwitchOpActor::CompileArrow(const std::unordered_map<void *, std::set<std::pair<AID, size_t>>> &receivers_map) {
  int ret = ModifySubgraphKernel();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ModifySubgraphKernel failed.";
    return ret;
  }

  ret = UpdateActorOutput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UpdateActorOutput failed.";
    return ret;
  }

  if (!kernel_->out_tensors().empty()) {
    CompileArrowThroughOutputTensors(receivers_map);
  }

  AppendOutputTensors();

  ret = CompileArrowThroughSwitchCall(receivers_map);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CompileArrowThroughSwitchCall failed.";
    return ret;
  }

  return ret;
}

int LiteSwitchOpActor::CreateSwitchTypeArrow(
  const std::unordered_map<void *, std::set<std::pair<AID, size_t>>> &receivers_map,
  const std::set<void *> &receiver_tensors, const Tensor *partial_in_tensor,
  std::vector<DataArrowPtr> *branch_output_data_arrows) {
  for (auto receiver_tensor : receiver_tensors) {
    MS_CHECK_TRUE_MSG(receivers_map.find(receiver_tensor) != receivers_map.end(), RET_ERROR,
                      "not find receiver_tensor in receivers_map");
    auto receiver_set = receivers_map.at(receiver_tensor);
    for (auto item : receiver_set) {
      for (size_t j = 0; j < kernel_->out_tensors().size(); ++j) {
        if (partial_in_tensor != kernel_->out_tensors()[j]) {
          continue;
        }
        auto arrow = std::make_shared<DataArrow>(j, item.first, item.second);
        MS_CHECK_TRUE_MSG(arrow != nullptr, RET_ERROR, "create data arrow failed.");
        branch_output_data_arrows->push_back(arrow);
        break;
      }
    }
  }
  return RET_OK;
}

int LiteSwitchOpActor::CompileArrowThroughSwitchCall(
  const std::unordered_map<void *, std::set<std::pair<AID, size_t>>> &receivers_map) {
  for (auto &partial_node : partial_nodes_) {
    if (partial_node == nullptr) {
      MS_LOG(ERROR) << "partial_node_ is nullptr.";
      return RET_NULL_PTR;
    }
    std::vector<DataArrowPtr> branch_output_data_arrows;
    auto partial_in_tensors = partial_node->in_tensors();
    for (size_t i = 0; i < partial_in_tensors.size(); ++i) {
      auto receiver_tensors = ctx_->GetLinkInfo(partial_in_tensors[i]);
      MS_CHECK_TRUE_MSG(!receiver_tensors.empty(), RET_ERROR, "no reviver for this actor");
      auto ret =
        CreateSwitchTypeArrow(receivers_map, receiver_tensors, partial_in_tensors[i], &branch_output_data_arrows);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "create switch type arrow failed, partial in tensor name: "
                      << partial_in_tensors[i]->tensor_name();
        return ret;
      }
    }
    all_branch_output_data_arrows_.push_back(branch_output_data_arrows);
  }
  return RET_OK;
}

int LiteSwitchOpActor::PrepareOutputData() {
  if (LiteOpActor::PrepareOutputData() != RET_OK) {
    MS_LOG(ERROR) << "lite actor prepare output data failed.";
    return RET_ERROR;
  }
  for (auto &branch_output_data_arrows : all_branch_output_data_arrows_) {
    std::vector<OpDataPtr<Tensor>> branch_outputs_data{};
    branch_outputs_data.resize(branch_output_data_arrows.size());
    for (size_t i = 0; i < branch_output_data_arrows.size(); i++) {
      auto &arrow = branch_output_data_arrows[i];
      auto data =
        std::make_shared<OpData<Tensor>>(this->GetAID(), (kernel_->out_tensors()).at(arrow->from_output_index_),
                                         static_cast<int>(arrow->to_input_index_));
      if (data == nullptr) {
        MS_LOG(ERROR) << "new branch output data failed.";
        return RET_NULL_PTR;
      }
      branch_outputs_data.at(i) = data;
    }
    all_branchs_output_data_.push_back(branch_outputs_data);
  }
  return RET_OK;
}

void LiteSwitchOpActor::DecreaseOtherBranchInputTensor(const size_t &index) {
  switch_type_node_->in_tensors()[kSwitchCondTensorIndex]->DecRefCount();
  for (size_t i = 0; i < partial_nodes_.size(); ++i) {
    if (i == index) {
      continue;
    }
    for (auto input : partial_nodes_[i]->in_tensors()) {
      input->DecRefCount();
    }
  }
}

void LiteSwitchOpActor::AsyncBranchOutput(const size_t &index, OpContext<Tensor> *context) {
  if (index >= all_branch_output_data_arrows_.size()) {
    MS_LOG(ERROR) << "index " << index
                  << " extend all_branch_output_data_arrows_.size(): " << all_branch_output_data_arrows_.size();
    context->SetFailed(RET_ERROR);
    return;
  }
  if (index >= all_branchs_output_data_.size()) {
    MS_LOG(ERROR) << "index " << index
                  << " extend all_branchs_output_data_.size(): " << all_branchs_output_data_.size();
    context->SetFailed(RET_ERROR);
    return;
  }
  auto branch_output_data_arrows = all_branch_output_data_arrows_.at(index);
  auto branch_outputs_data = all_branchs_output_data_.at(index);
  if (branch_output_data_arrows.size() != branch_outputs_data.size()) {
    MS_LOG(ERROR) << "index " << index
                  << " extend all_branchs_output_data_.size(): " << all_branchs_output_data_.size();
    context->SetFailed(RET_ERROR);
    return;
  }
  for (size_t i = 0; i < branch_output_data_arrows.size(); ++i) {
    auto &data = branch_outputs_data.at(i);
    Async(branch_output_data_arrows[i]->to_op_id_, get_actor_mgr(), &mindspore::OpActor<Tensor>::RunOpData, data.get(),
          context);
  }
}

void LiteSwitchOpActor::RunOpData(OpData<Tensor> *inputs, OpContext<Tensor> *context) {
  auto op_uuid = context->sequential_num_;
  input_op_datas_[op_uuid].push_back(inputs);
  inputs_data_[inputs->index_] = inputs->data_;
  if (input_op_datas_[op_uuid].size() < kernel_->in_tensors().size()) {
    return;
  }

  auto ret = InitInputData();
  if (ret != RET_OK) {
    input_op_datas_.erase(op_uuid);
    context->SetFailed(ret);
    return;
  }

  ret = RunKernel(*(reinterpret_cast<const KernelCallBack *>(context->kernel_call_back_before_)),
                  *(reinterpret_cast<const KernelCallBack *>(context->kernel_call_back_after_)));
  if (ret != RET_OK) {
    input_op_datas_.erase(op_uuid);
    context->SetFailed(ret);
    return;
  }
  input_op_datas_.erase(op_uuid);

  auto cond_ptr = reinterpret_cast<bool *>(switch_type_node_->in_tensors()[kSwitchCondTensorIndex]->data());
  if (cond_ptr == nullptr) {
    MS_LOG(ERROR) << "switch cond input data is nullptr.";
    context->SetFailed(RET_NULL_PTR);
    return;
  }
  size_t index = static_cast<size_t>(*cond_ptr);
  DecreaseOtherBranchInputTensor(index);
  AsyncBranchOutput(index, context);
  if (!output_data_arrows_.empty()) {
    AsyncOutput(context);
    SetOutputData(context);
  }
}
}  // namespace mindspore::lite
