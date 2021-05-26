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

#include <utility>
#include "src/lite_mindrt.h"
#include "mindrt/include/mindrt.hpp"
#include "src/lite_kernel_util.h"
#include "nnacl/partial_fusion_parameter.h"
#include "src/common/tensor_util.h"
#include "nnacl/base/cast_base.h"

namespace mindspore::lite {

int LiteOpActor::CompileArrowThroughOutputKernels() {
  output_op_arrows_.clear();
  int out_tensor_size = static_cast<int>(kernel_->out_tensors().size());
  for (int i = 0; i < out_tensor_size; i++) {
    for (auto out : kernel_->out_kernels()) {
      int in_tensor_size = static_cast<int>(out->in_tensors().size());
      int to_input_index = -1;
      for (int j = 0; j < in_tensor_size; j++) {
        if (kernel_->out_tensors()[i] == out->in_tensors()[j]) {
          to_input_index = j;
          break;
        }
      }
      if (to_input_index == -1) {
        continue;
      }
      auto id = out->name() + this->GetAID().Url();
      auto arrow = std::make_shared<OpArrow>(i, AID(id), to_input_index);
      if (arrow == nullptr) {
        MS_LOG(ERROR) << "create OpArrow failed, out kernel: " << out->name();
        return RET_ERROR;
      }
      output_op_arrows_.emplace_back(std::move(arrow));
    }
  }
  return RET_OK;
}

int LiteOpActor::CompileArrowThroughPartialCall() {
  auto *subgraph_kernel = reinterpret_cast<kernel::SubGraphKernel *>(kernel_);
  if (subgraph_kernel == nullptr) {
    MS_LOG(INFO) << "kernel is not subgraph kernel, no partial call.";
    return RET_OK;
  }
  for (auto &node : subgraph_kernel->nodes()) {
    if (node->type() != schema::PrimitiveType_Call) {
      continue;
    }
    call_node_ = node;
    auto partial_node = kernel::LiteKernelUtil::GetInputsSpecificNode(node, schema::PrimitiveType_PartialFusion);
    if (!partial_node) {
      continue;
    }
    partial_node_ = partial_node;

    auto partial_para = reinterpret_cast<PartialParameter *>(partial_node->op_parameter());
    auto out_actor_id = subgraph_index_to_actor.at(partial_para->sub_graph_index_);
    kernel_->set_out_tensors(partial_node->in_tensors());
    for (size_t i = 0; i < partial_node->in_tensors().size(); ++i) {
      auto arrow = std::make_shared<OpArrow>(i, out_actor_id, i);
      if (arrow == nullptr) {
        MS_LOG(ERROR) << "create OpArrow failed";
        return RET_ERROR;
      }
      output_op_arrows_.emplace_back(std::move(arrow));
    }
  }

  subgraph_kernel->DropNode(partial_node_);
  subgraph_kernel->DropNode(call_node_);
  return RET_OK;
}

int LiteOpActor::CompileArrow() {
  output_op_arrows_.clear();
  int ret = CompileArrowThroughPartialCall();
  if (ret != RET_OK) {
    output_op_arrows_.clear();
    MS_LOG(ERROR) << "CompileArrowThroughPartialCall failed.";
    return ret;
  }
  if (!output_op_arrows_.empty()) {
    MS_LOG(INFO) << "CompileArrowThroughPartialCall done.";
    return RET_OK;
  }
  ret = CompileArrowThroughOutputKernels();
  if (ret != RET_OK) {
    output_op_arrows_.clear();
    MS_LOG(ERROR) << "CompileArrowThroughOutputKernels failed.";
    return ret;
  }
  return ret;
}

int LiteOpActor::CheckInputData() {
  if (kernel_->in_tensors().size() != inputs_data_.size()) {
    MS_LOG(ERROR) << "kernel:" << kernel_->name() << "inputs_data_.size(): " << inputs_data_.size()
                  << " vs kernel_->in_tensors().size(): " << kernel_->in_tensors().size() << " are not equal.";
    return RET_PARAM_INVALID;
  }

  for (size_t i = 0; i < inputs_data_.size(); ++i) {
    if (kernel_->in_tensors()[i]->shape() != inputs_data_[i]->shape()) {
      MS_LOG(ERROR) << "inputs_data_[" << i << "].shape: " << inputs_data_[i]->shape() << " vs kernel_->in_tensors()["
                    << i << "].shape: " << kernel_->in_tensors()[i]->shape() << " are not equal.";
      return RET_PARAM_INVALID;
    }
  }
  return RET_OK;
}

void LiteOpActor::MoveInputData(Tensor *dst_tensor, Tensor *src_tensor) {
  memcpy(dst_tensor->MutableData(), src_tensor->data_c(), src_tensor->Size());
  dst_tensor->IncRefCount();
  src_tensor->DecRefCount();
}
void LiteOpActor::CopyInputData(Tensor *dst_tensor, Tensor *src_tensor) {
  CastTensorData(dst_tensor, src_tensor);
  dst_tensor->IncRefCount();
  src_tensor->DecRefCount();
}

int LiteOpActor::CastTensorData(Tensor *dst, Tensor *src) {
  if (dst->shape() != src->shape()) {
    MS_LOG(ERROR) << "dst tensor: " << dst->tensor_name() << " shape: " << dst->shape() << " vs "
                  << "src tensor: " << src->tensor_name() << " shape: " << src->shape();
    return RET_PARAM_INVALID;
  }
  auto dst_data = dst->MutableData();
  auto src_data = src->MutableData();
  auto src_nums_size = src->ElementsNum();
  auto dst_data_type = static_cast<int>(dst->data_type());
  auto src_data_type = static_cast<int>(src->data_type());

  if (dst_data_type == kNumberTypeFloat32 && src_data_type == kNumberTypeFloat16) {
    Fp16ToFloat32(static_cast<uint16_t *>(src_data), static_cast<float *>(dst_data), src_nums_size);
  } else if (dst_data_type == kNumberTypeFloat16 && src_data_type == kNumberTypeFloat32) {
    Float32ToFp16(static_cast<float *>(src_data), static_cast<uint16_t *>(dst_data), src_nums_size);
  } else {
    MS_LOG(ERROR) << "not support dst_data_type: " << dst_data_type << " src_data_type: " << src_data_type;
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int LiteOpActor::SetInputData() {
  for (size_t i = 0; i < inputs_data_.size(); ++i) {
    auto dst_tensor = kernel_->in_tensors()[i];
    auto src_tensor = inputs_data_[i];
    if (src_tensor->data_type() != dst_tensor->data_type()) {
      CopyInputData(dst_tensor, src_tensor);
    } else {
      MoveInputData(dst_tensor, src_tensor);
    }
  }
  return RET_OK;
}

void LiteOpActor::AsyncOutput(OpContext<Tensor> *context) {
  for (const auto &op_arrow : output_op_arrows_) {
    auto data = outputs_data_.at(op_arrow->from_output_index_);
    Async(op_arrow->to_op_id_, &mindspore::OpActor<Tensor>::RunOpData, data, context);
  }
}

void LiteOpActor::AddResultIndex(size_t index) { results_index_.push_back(index); }

void LiteOpActor::SetOutputData(OpContext<Tensor> *context) {
  for (auto index : results_index_) {
    context->SetResult(index, RET_OK);
  }
}

int LiteOpActor::PrepareOutputData() {
  for (auto &arrow : output_op_arrows_) {
    auto data = std::make_shared<OpData<Tensor>>(arrow->to_op_id_, kernel_->out_tensors().at(arrow->from_output_index_),
                                                 static_cast<int>(arrow->to_input_index_));
    outputs_data_.emplace_back(data);
  }
  return RET_OK;
}

std::vector<std::shared_ptr<LiteOpActor>> CreateOpActor(const std::vector<kernel::LiteKernel *> &kernels) {
  std::vector<std::shared_ptr<LiteOpActor>> actors;
  std::unordered_map<size_t, AID> partial_map{};
  auto thread_pool = kernels[0]->Context()->thread_pool_;
  if (thread_pool == nullptr) {
    MS_LOG(ERROR) << "thread pool is nullptr";
    return actors;
  }
  for (size_t i = 0; i < kernels.size(); ++i) {
    if ((kernel::LiteKernelUtil::IsSwitchCall(kernels[i]))) {
      auto switch_actor = std::make_shared<LiteSwitchOpActor>(kernels[i]);
      if (switch_actor == nullptr) {
        MS_LOG(ERROR) << "create LiteSwitchOpActor failed: " << kernels[i]->name();
        actors.clear();
        return actors;
      }
      switch_actor->set_thread_pool(thread_pool);
      partial_map[i] = switch_actor->GetAID();
      actors.push_back(switch_actor);
    } else {
      auto actor = std::make_shared<LiteOpActor>(kernels[i]);
      if (actor == nullptr) {
        MS_LOG(ERROR) << "create LiteOpActor failed: " << kernels[i]->name();
        actors.clear();
        return actors;
      }
      actor->set_thread_pool(thread_pool);
      partial_map[i] = actor->GetAID();
      actors.push_back(actor);
    }
  }

  for (auto &actor : actors) {
    actor->SetPartialMap(partial_map);
    auto aid = mindspore::Spawn(actor);
  }
  return actors;
}

int LiteSwitchOpActor::CompileTrueBranchArrow() {
  true_branch_output_op_arrows_.clear();
  if (true_partial_node_ == nullptr) {
    MS_LOG(ERROR) << "true_partial_node_ is nullptr.";
    return RET_NULL_PTR;
  }
  auto true_partial_para = reinterpret_cast<PartialParameter *>(true_partial_node_->op_parameter());
  if (true_partial_para == nullptr) {
    MS_LOG(ERROR) << "true_partial_node_->op_parameter() is nullptr.";
    return RET_NULL_PTR;
  }
  auto true_branch_actor_id = subgraph_index_to_actor.at(true_partial_para->sub_graph_index_);

  for (size_t i = 0; i < true_partial_node_->in_tensors().size(); ++i) {
    int out_tensor_size = static_cast<int>(kernel_->out_tensors().size());
    for (int j = 0; j < out_tensor_size; ++j) {
      if (true_partial_node_->in_tensors()[i] != kernel_->out_tensors()[j]) {
        continue;
      }
      auto arrow = std::make_shared<OpArrow>(j, true_branch_actor_id, i);
      if (arrow == nullptr) {
        MS_LOG(ERROR) << "create OpArrow failed";
        return RET_ERROR;
      }
      true_branch_output_op_arrows_.emplace_back(std::move(arrow));
    }
  }
  return RET_OK;
}

int LiteSwitchOpActor::CompileFalseBranchArrow() {
  false_branch_output_op_arrows_.clear();
  if (false_partial_node_ == nullptr) {
    MS_LOG(ERROR) << "false_partial_node_ is nullptr.";
    return RET_NULL_PTR;
  }
  auto false_partial_para = reinterpret_cast<PartialParameter *>(false_partial_node_->op_parameter());
  if (false_partial_para == nullptr) {
    MS_LOG(ERROR) << "false_partial_para->op_parameter() is nullptr.";
    return RET_NULL_PTR;
  }
  auto false_branch_actor_id = subgraph_index_to_actor.at(false_partial_para->sub_graph_index_);

  for (size_t i = 0; i < false_partial_node_->in_tensors().size(); ++i) {
    int out_tensor_size = static_cast<int>(kernel_->out_tensors().size());
    for (int j = 0; j < out_tensor_size; ++j) {
      if (false_partial_node_->in_tensors()[i] != kernel_->out_tensors()[j]) {
        continue;
      }
      auto arrow = std::make_shared<OpArrow>(j, false_branch_actor_id, i);
      if (arrow == nullptr) {
        MS_LOG(ERROR) << "create OpArrow failed";
        return RET_ERROR;
      }
      false_branch_output_op_arrows_.emplace_back(std::move(arrow));
    }
  }
  return RET_OK;
}

int LiteSwitchOpActor::GetSwitchAndCallNode(kernel::SubGraphKernel *subgraph_kernel) {
  for (auto &node : subgraph_kernel->nodes()) {
    if (node->type() != schema::PrimitiveType_Call) {
      continue;
    }
    call_node_ = node;
    auto switch_node = kernel::LiteKernelUtil::GetInputsSpecificNode(node, schema::PrimitiveType_Switch);
    if (!switch_node) {
      continue;
    }
    switch_node_ = switch_node;
    if (switch_node->in_kernels().size() != kSwitchInputsSize) {
      MS_LOG(ERROR) << "switch input size: " << switch_node->in_kernels().size();
      return RET_MEMORY_FAILED;
    }

    bool_node_ = switch_node->in_kernels().at(kSwitchCondInputIndex);
    true_partial_node_ = switch_node->in_kernels().at(kSwitchTruePartialInputIndex);
    false_partial_node_ = switch_node->in_kernels().at(kSwitchFalsePartialInputIndex);
    break;
  }
  return RET_OK;
}

void LiteSwitchOpActor::AppendOutputTensors() {
  output_tensors_.push_back(bool_node_->out_tensors().front());
  for (auto &tensor : true_partial_node_->in_tensors()) {
    if (std::find(output_tensors_.begin(), output_tensors_.end(), tensor) == output_tensors_.end()) {
      output_tensors_.push_back(tensor);
    }
  }
  for (auto &tensor : false_partial_node_->in_tensors()) {
    if (std::find(output_tensors_.begin(), output_tensors_.end(), tensor) == output_tensors_.end()) {
      output_tensors_.push_back(tensor);
    }
  }
  kernel_->set_out_tensors(output_tensors_);
}

int LiteSwitchOpActor::CompileArrowThroughSwitchCall() {
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

  AppendOutputTensors();

  ret = CompileTrueBranchArrow();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CompileTrueBranchArrow failed.";
    true_branch_output_op_arrows_.clear();
    return ret;
  }

  ret = CompileFalseBranchArrow();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CompileFalseBranchArrow failed.";
    false_branch_output_op_arrows_.clear();
    true_branch_output_op_arrows_.clear();
    return ret;
  }

  subgraph_kernel->DropNode(call_node_);
  subgraph_kernel->DropNode(switch_node_);
  subgraph_kernel->DropNode(true_partial_node_);
  subgraph_kernel->DropNode(false_partial_node_);

  return ret;
}

int LiteSwitchOpActor::CompileArrow() {
  int ret = CompileArrowThroughSwitchCall();
  if (ret != RET_OK) {
    true_branch_output_op_arrows_.clear();
    false_branch_output_op_arrows_.clear();
    MS_LOG(ERROR) << "CompileArrowThroughSwitchCall failed.";
    return ret;
  }
  if (!true_branch_output_op_arrows_.empty() && !false_branch_output_op_arrows_.empty()) {
    MS_LOG(INFO) << "CompileArrowThroughSwitchCall done.";
    return RET_OK;
  }
  ret = CompileArrowThroughOutputKernels();
  if (ret != RET_OK) {
    output_op_arrows_.clear();
    MS_LOG(ERROR) << "CompileArrowThroughOutputKernels failed.";
    return ret;
  }
  return ret;
}

int LiteSwitchOpActor::PrepareOutputData() {
  for (auto &arrow : true_branch_output_op_arrows_) {
    auto data = std::make_shared<OpData<Tensor>>(arrow->to_op_id_, kernel_->out_tensors().at(arrow->from_output_index_),
                                                 static_cast<int>(arrow->to_input_index_));
    true_branch_outputs_data_.emplace_back(data);
  }

  for (auto &arrow : false_branch_output_op_arrows_) {
    auto data = std::make_shared<OpData<Tensor>>(arrow->to_op_id_, kernel_->out_tensors().at(arrow->from_output_index_),
                                                 static_cast<int>(arrow->to_input_index_));
    false_branch_outputs_data_.emplace_back(data);
  }
  return RET_OK;
}

void LiteSwitchOpActor::AsyncTrueBranchOutput(OpContext<Tensor> *context) {
  MS_ASSERT(true_branch_output_op_arrows_.size() == true_branch_outputs_data_.size());
  for (size_t i = 0; i < true_branch_output_op_arrows_.size(); ++i) {
    auto &data = true_branch_outputs_data_.at(i);
    Async(true_branch_output_op_arrows_[i]->to_op_id_, &mindspore::OpActor<Tensor>::RunOpData, data, context);
  }
}

void LiteSwitchOpActor::AsyncFalseBranchOutput(OpContext<Tensor> *context) {
  MS_ASSERT(false_branch_output_op_arrows_.size() == false_branch_outputs_data_.size());
  for (size_t i = 0; i < false_branch_output_op_arrows_.size(); ++i) {
    auto &data = false_branch_outputs_data_.at(i);
    Async(false_branch_output_op_arrows_[i]->to_op_id_, &mindspore::OpActor<Tensor>::RunOpData, data, context);
  }
}

int MindrtInit() { return mindspore::Initialize("tcp://127.0.0.1:8080", "", "", "", 1); }

void MindrtTerminate(const std::vector<std::shared_ptr<LiteOpActor>> &actor_list) {
  for (const auto &actor : actor_list) {
    mindspore::Terminate(actor->GetAID());
  }
}

}  // namespace mindspore::lite
