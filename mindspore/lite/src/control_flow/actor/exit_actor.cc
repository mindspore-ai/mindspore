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

#include "src/control_flow/actor/exit_actor.h"
#include <algorithm>
#include "src/control_flow/kernel/exit_subgraph_kernel.h"
#include "src/runtime/kernel_exec_util.h"
#include "src/common/tensor_util.h"

namespace {
const constexpr int kEntranceTensorIndex = 0;
}
namespace mindspore::lite {
void LiteExitOpActor::RunOpData(OpData<Tensor> *inputs, OpContext<Tensor> *context) {
  auto op_uuid = context->sequential_num_;
  input_op_datas_[op_uuid].push_back(inputs);
  inputs_data_[inputs->index_] = inputs->data_;
  SetEntranceInputAID(inputs);
  if (input_op_datas_[op_uuid].size() < kernel_->in_tensors().size()) {
    return;
  }

  auto ret = InitInputData();
  input_op_datas_.erase(op_uuid);
  if (ret != RET_OK) {
    context->SetFailed(ret);
    return;
  }
  AsyncOutput(context);
  return;
}

int LiteExitOpActor::InitInputData() {
  auto ret = SetInputShape();

  for (size_t i = 1; i < inputs_data_.size(); ++i) {
    auto dst_tensor = kernel_->out_tensors()[i - 1];
    auto src_tensor = inputs_data_[i];
    dst_tensor->set_data_type(src_tensor->data_type());
    if (src_tensor->allocator() == nullptr || src_tensor->IsGraphInput()) {
      (void)SetTensorData(dst_tensor, src_tensor);
    } else {
      (void)MoveTensorData(dst_tensor, src_tensor);
    }
  }
  return ret;
}

int LiteExitOpActor::SetInputShape() {
  auto ret = RET_OK;
  for (size_t i = 1; i < inputs_data_.size(); ++i) {
    auto &output_tensor = kernel_->out_tensors()[i - 1];
    if (output_tensor->shape() == inputs_data_[i]->shape()) {
      continue;
    }
    ret = SetTensorShape(output_tensor, inputs_data_[i]);
    MS_CHECK_FALSE_MSG(ret != RET_OK, ret, "set input shape failed.");
  }
  return RET_OK;
}

void LiteExitOpActor::SetEntranceInputAID(OpData<Tensor> *inputs) {
  if (inputs->index_ == kEntranceTensorIndex) {
    entrance_input_aid_ = inputs->op_id_;
  }
}

int LiteExitOpActor::PrepareOutputData() {
  // exit actor has not calculating, so send input directly.
  outputs_data_.resize(output_data_arrows_.size());
  for (size_t i = 0; i < output_data_arrows_.size(); i++) {
    auto &arrow = output_data_arrows_[i];
    auto data = std::make_shared<OpData<Tensor>>(this->GetAID(), (kernel_->out_tensors()).at(arrow->from_output_index_),
                                                 static_cast<int>(arrow->to_input_index_));
    if (data == nullptr) {
      MS_LOG(ERROR) << "new output_data failed.";
      return RET_NULL_PTR;
    }
    outputs_data_.at(i) = data;
  }
  return RET_OK;
}

void LiteExitOpActor::AsyncOutput(OpContext<Tensor> *context) {
  AID to_op_id;
  bool find_to_op_aid = false;
  for (auto info : all_mapping_info_) {
    if (info.partial_input_aid == entrance_input_aid_) {
      find_to_op_aid = true;
      to_op_id = info.call_output_aid;
    }
  }

  if (!find_to_op_aid) {
    MS_LOG(ERROR) << "exit actor can not find output actor.";
    context->SetFailed(RET_ERROR);
    return;
  }

  if (to_op_id.Name() == "") {
    SetOutputData(context);
  }

  for (size_t i = 0; i < output_data_arrows_.size(); i++) {
    if (output_data_arrows_[i]->to_op_id_ != to_op_id && output_data_arrows_[i]->to_op_id_.Name() != "") {
      continue;
    }
    auto data = outputs_data_.at(i);
    Async(to_op_id, get_actor_mgr(), &mindspore::OpActor<Tensor>::RunOpData, data.get(), context);
  }
}

int LiteExitOpActor::PreInit(std::vector<std::shared_ptr<LiteOpActor>> *actors,
                             std::unordered_map<Tensor *, Tensor *> *input_map) {
  auto ret = IsolateInputData(actors, input_map);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "isolate input data failed.";
    return ret;
  }

  ret = CreateMappingInfo();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "create partial call pairs failed.";
    return ret;
  }

  ret = RecordCallNodeOutputActor(actors);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "record call node outputs AIDs failed";
    return ret;
  }
  return RET_OK;
}

bool LiteExitOpActor::IsSubSet(const std::vector<lite::Tensor *> &all_set, const std::vector<lite::Tensor *> &sub_set) {
  if (sub_set.size() > all_set.size()) {
    return false;
  }
  for (auto &sub_item : sub_set) {
    if (std::find(all_set.begin(), all_set.end(), sub_item) == all_set.end()) {
      return false;
    }
  }
  return true;
}

int LiteExitOpActor::RecordCallNodeOutputActor(std::vector<std::shared_ptr<LiteOpActor>> *actors) {
  actors_ = actors;
  for (auto actor : *actors_) {
    auto actor_in_tensors = actor->GetKernel()->in_tensors();
    for (auto &info : all_mapping_info_) {
      auto &call = info.call_node;
      if (IsSubSet(actor_in_tensors, call->out_tensors())) {
        info.call_output_aid = actor->GetAID();
      }
    }
  }
  return RET_OK;
}

int LiteExitOpActor::CreateMappingInfo() {
  auto exit_subgraph_kernel = reinterpret_cast<kernel::ExitSubGraphKernel *>(kernel_);
  if (exit_subgraph_kernel == nullptr) {
    MS_LOG(ERROR) << "cast to exit kernel failed.";
    return RET_ERROR;
  }
  auto partial_set = exit_subgraph_kernel->GetPartials();
  for (auto partial : partial_set) {
    auto call_node = kernel::KernelExecUtil::GetPartialOutputCall(partial);
    if (call_node == nullptr) {
      MS_LOG(ERROR) << "get partial node: " << partial->name() << " 's call output node failed.";
      return RET_ERROR;
    }
    MappingInfo info(partial, call_node);
    all_mapping_info_.emplace_back(info);
  }
  return RET_OK;
}

int LiteExitOpActor::PostInit() {
  auto ret = PrepareOutputData();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "prepare output data failed.";
    return ret;
  }

  RecordPartialNodeInputActor();
  return RET_OK;
}

void LiteExitOpActor::RecordPartialNodeInputActor() {
  for (auto actor : *actors_) {
    auto actor_partial_nodes = actor->GetPartialKernels();
    if (actor_partial_nodes.empty()) {
      continue;
    }
    for (auto &info : all_mapping_info_) {
      auto partial = info.partial_node;
      if (actor_partial_nodes.find(partial) == actor_partial_nodes.end()) {
        continue;
      }
      info.partial_input_aid = actor->GetAID();
    }
  }
}
}  // namespace mindspore::lite
