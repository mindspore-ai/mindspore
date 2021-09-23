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
#include <algorithm>
#include "src/lite_mindrt.h"
#include "mindrt/include/mindrt.hpp"
#include "src/lite_kernel_util.h"
#include "src/common/tensor_util.h"
#include "src/runtime/inner_allocator.h"
#include "src/runtime/kernel/arm/base/partial_fusion.h"
#ifdef ENABLE_FP16
#include "src/runtime/kernel/arm/fp16/fp16_op_handler.h"
#endif

namespace mindspore::lite {
void LiteOpActor::RunOpData(OpData<lite::Tensor> *inputs, OpContext<lite::Tensor> *context) {
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
  AsyncOutput(context);

  SetOutputData(context);

  return;
}

bool OfflineIsolated(const std::vector<kernel::LiteKernel *> &kernels, const kernel::LiteKernel &this_kernel,
                     const lite::Tensor &this_input_tensor) {
  if (this_input_tensor.IsGraphInput()) {
    return false;
  }
  for (auto &kernel : kernels) {
    if (kernel == &this_kernel) {
      continue;
    }
    if (std::any_of(kernel->out_tensors().begin(), kernel->out_tensors().end(),
                    [&this_input_tensor](lite::Tensor *tensor) { return tensor == &this_input_tensor; })) {
      return false;
    }
  }
  return true;
}

void LiteOpActor::ReplaceNodeInTensor(kernel::LiteKernel *kernel, Tensor *old_tensor, Tensor *new_tensor) {
  int ref_count = 0;
#ifndef DELEGATE_CLIP
  /* set op input for calculate */
  if (kernel->desc().arch == kernel::kDelegate) {
    ref_count++;
  } else {
#endif
    for (auto in_node : reinterpret_cast<kernel::SubGraphKernel *>(kernel)->in_nodes()) {
      for (size_t node_in_index = 0; node_in_index < in_node->in_tensors().size(); node_in_index++) {
        if (old_tensor == in_node->in_tensors()[node_in_index]) {
          in_node->set_in_tensor(new_tensor, node_in_index);
          ref_count++;
        }
      }
    }
#ifndef DELEGATE_CLIP
  }
#endif
  new_tensor->set_init_ref_count(ref_count);
}

int LiteOpActor::IsolateInputData(std::vector<std::shared_ptr<LiteOpActor>> *actors) {
  std::vector<kernel::LiteKernel *> kernels{};
  std::transform(actors->begin(), actors->end(), std::back_inserter(kernels),
                 [](std::shared_ptr<LiteOpActor> actor) { return actor->kernel_; });
  size_t in_tensor_size = kernel_->in_tensors().size();
  for (size_t i = 0; i < in_tensor_size; i++) {
    Tensor *old_tensor = kernel_->in_tensors()[i];

    if (OfflineIsolated(kernels, *kernel_, *old_tensor)) {
      if (old_tensor->data_type() == kNumberTypeFloat16 || old_tensor->data_type() == kNumberTypeFloat32) {
        old_tensor->set_data_type(kernel_->desc().data_type);
      }
#ifndef CONTROLFLOW_TENSORLIST_CLIP
      if (old_tensor->data_type() == kObjectTypeTensorType) {
        auto old_tensorlist = reinterpret_cast<TensorList *>(old_tensor);
        if (old_tensorlist->tensors_data_type() == kNumberTypeFloat16 ||
            old_tensorlist->tensors_data_type() == kNumberTypeFloat32) {
          old_tensorlist->set_tensors_data_type(kernel_->desc().data_type);
        }
      }
#endif
      old_tensor->set_allocator(kernel_->Context()->allocator);
      continue;
    }

    TypeId new_data_type = old_tensor->data_type();
    if (old_tensor->data_type() == kNumberTypeFloat16 || old_tensor->data_type() == kNumberTypeFloat32) {
      new_data_type = kernel_->desc().data_type;
    }

    Tensor *new_tensor = new Tensor(new_data_type, old_tensor->shape(), old_tensor->format(), old_tensor->category());
    if (new_tensor == nullptr) {
      MS_LOG(ERROR) << "new Tensor failed.";
      return RET_NULL_PTR;
    }
    new_tensor->set_allocator(old_tensor->allocator());
    if (new_tensor->allocator() == nullptr && kernel_->Context() != nullptr &&
        kernel_->desc().arch != kernel::kDelegate) {
      new_tensor->set_allocator(kernel_->Context()->allocator);
    }

    new_tensor->set_tensor_name(kernel_->name() + "_duplicate_" + old_tensor->tensor_name());
    for (LiteQuantParam quant : old_tensor->quant_params()) {
      new_tensor->AddQuantParam(quant);
    }
    isolate_input_map_.insert(std::make_pair(new_tensor, old_tensor));
    ReplaceNodeInTensor(kernel_, old_tensor, new_tensor);
    /* set subgraph input for copy data */
    kernel_->set_in_tensor(new_tensor, i);
  }
  return RET_OK;
}

int LiteOpActor::LiteActorInit(std::vector<std::shared_ptr<LiteOpActor>> *actors) {
  /* Init output arrow */
  auto ret = CompileArrow();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "compile arrow failed.";
    return ret;
  }

  /* Init Actor output data */
  ret = PrepareOutputData();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "prepare output data failed.";
    return ret;
  }

  /* subgraph transaction isolation */
  ret = IsolateInputData(actors);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "isolate input data failed.";
    return ret;
  }
  return RET_OK;
}

int LiteOpActor::ResizeGraphInput(const std::vector<mindspore::tensor::MSTensor *> &inputs,
                                  const std::vector<std::vector<int>> &dims) {
  for (auto map : isolate_input_map_) {
    auto isolate_tensor = map.first;
    auto src_tensor = map.second;
    for (size_t i = 0; i < inputs.size(); i++) {
      if (src_tensor == inputs[i]) {
        isolate_tensor->set_shape(dims[i]);
      }
    }
  }
  return RET_OK;
}

int LiteOpActor::CompileArrowThroughOutputKernels() {
  output_data_arrows_.clear();
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
      auto arrow = std::make_shared<DataArrow>(i, AID(id), to_input_index);
      if (arrow == nullptr) {
        MS_LOG(ERROR) << "create DataArrow failed, out kernel: " << out->name();
        return RET_ERROR;
      }
      output_data_arrows_.emplace_back(std::move(arrow));
    }
  }
  return RET_OK;
}

#ifndef CONTROLFLOW_TENSORLIST_CLIP
int LiteOpActor::CompileArrowThroughPartialCall() {
#ifndef DELEGATE_CLIP
  if (kernel_->desc().arch == kernel::kDelegate) {
    MS_LOG(INFO) << "kernel is delegate subgraph kernel.";
    return RET_OK;
  }
#endif
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
    auto subgraph = reinterpret_cast<kernel::PartialFusionKernel *>(partial_node->kernel())->subgraph_kernel();
    auto out_actor_id = subgraph_to_actor_.at(subgraph);

    kernel_->set_out_tensors(partial_node->in_tensors());
    for (size_t i = 0; i < partial_node->in_tensors().size(); ++i) {
      auto arrow = std::make_shared<DataArrow>(i, out_actor_id, i);
      if (arrow == nullptr) {
        MS_LOG(ERROR) << "create DataArrow failed";
        return RET_ERROR;
      }
      output_data_arrows_.emplace_back(std::move(arrow));
    }
  }

  subgraph_kernel->DropNode(partial_node_);
  subgraph_kernel->DropNode(call_node_);
  return RET_OK;
}
#endif

int LiteOpActor::CompileArrow() {
  int ret;
  output_data_arrows_.clear();
#ifndef CONTROLFLOW_TENSORLIST_CLIP
  ret = CompileArrowThroughPartialCall();
  if (ret != RET_OK) {
    output_data_arrows_.clear();
    MS_LOG(ERROR) << "CompileArrowThroughPartialCall failed.";
    return ret;
  }
  if (!output_data_arrows_.empty()) {
    MS_LOG(INFO) << "CompileArrowThroughPartialCall done.";
    return RET_OK;
  }
#endif
  ret = CompileArrowThroughOutputKernels();
  if (ret != RET_OK) {
    output_data_arrows_.clear();
    MS_LOG(ERROR) << "CompileArrowThroughOutputKernels failed.";
    return ret;
  }
  return ret;
}

void LiteOpActor::MoveTensorInputData(Tensor *dst_tensor, Tensor *src_tensor) {
  MS_ASSERT(src_tensor != dst_tensor);
  dst_tensor->FreeData();
  dst_tensor->ResetRefCount();
  dst_tensor->set_allocator(src_tensor->allocator());

  src_tensor->allocator()->IncRefCount(src_tensor->data(), dst_tensor->ref_count());

  if (src_tensor->data() != nullptr) {
    dst_tensor->set_data(src_tensor->MutableData()); /* using MutableData to sync GPU data */
  }

  dst_tensor->set_own_data(src_tensor->own_data());
  src_tensor->DecRefCount();
}

void LiteOpActor::MoveInputData(Tensor *dst_tensor, Tensor *src_tensor) {
  if (src_tensor == dst_tensor) {
    MS_LOG(INFO) << "no need to move.";
    return;
  }
  MS_ASSERT(src_tensor->allocator() != nullptr);
#ifndef CONTROLFLOW_TENSORLIST_CLIP
  if (src_tensor->data_type() == kObjectTypeTensorType) {
    MoveTensorListInputData(reinterpret_cast<TensorList *>(dst_tensor), reinterpret_cast<TensorList *>(src_tensor));
  } else {
    MoveTensorInputData(dst_tensor, src_tensor);
  }
#else
  MoveTensorInputData(dst_tensor, src_tensor);
#endif
  return;
}

void LiteOpActor::SetInputData(Tensor *dst_tensor, Tensor *src_tensor) {
  dst_tensor->set_data(src_tensor->data());
  dst_tensor->set_own_data(false);
}

int LiteOpActor::CastInputData(Tensor *dst, Tensor *src) {
  int ret = RET_OK;
#ifndef CONTROLFLOW_TENSORLIST_CLIP
  if (src->data_type() != kObjectTypeTensorType) {
    ret = CastTensorInputData(dst, src);
  } else {
    ret = CastTensorListInputData(reinterpret_cast<TensorList *>(dst), reinterpret_cast<TensorList *>(src));
  }
#else
  ret = CastTensorInputData(dst, src);
#endif
  src->DecRefCount();
  return ret;
}

bool LiteOpActor::NeedCastData(Tensor *dst_tensor, Tensor *src_tensor) {
  if (dst_tensor->data_type() != kObjectTypeTensorType && src_tensor->data_type() != kObjectTypeTensorType &&
      dst_tensor->data_type() != src_tensor->data_type()) {
    return true;
  }
#ifndef CONTROLFLOW_TENSORLIST_CLIP
  if (dst_tensor->data_type() == kObjectTypeTensorType && src_tensor->data_type() == kObjectTypeTensorType &&
      reinterpret_cast<TensorList *>(dst_tensor)->tensors_data_type() !=
        reinterpret_cast<TensorList *>(src_tensor)->tensors_data_type()) {
    return true;
  }
#endif
  return false;
}

int LiteOpActor::CastTensorInputData(Tensor *dst, Tensor *src) {
  dst->MallocData();
  dst->ResetRefCount();
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
  if (dst->shape() != src->shape()) {
    MS_LOG(ERROR) << "dst tensor: " << dst->tensor_name() << " shape: " << dst->shape() << " vs "
                  << "src tensor: " << src->tensor_name() << " shape: " << src->shape();
    return RET_PARAM_INVALID;
  }
  auto dst_data = dst->MutableData(); /* using MutableData to sync GPU data */
  auto src_data = src->MutableData();
  auto src_nums_size = src->ElementsNum();
  auto dst_data_type = static_cast<int>(dst->data_type());
  auto src_data_type = static_cast<int>(src->data_type());
  if (dst_data_type == kNumberTypeFloat32 && src_data_type == kNumberTypeFloat16) {
    Float16ToFloat32_fp16_handler(src_data, dst_data, src_nums_size, support_fp16_);
  } else if (dst_data_type == kNumberTypeFloat16 && src_data_type == kNumberTypeFloat32) {
    Float32ToFloat16_fp16_handler(src_data, dst_data, src_nums_size, support_fp16_);
  } else {
    MS_LOG(ERROR) << "not support dst_data_type: " << dst_data_type << " src_data_type: " << src_data_type;
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
#endif
  return RET_ERROR;
}

#ifndef CONTROLFLOW_TENSORLIST_CLIP
void LiteOpActor::MoveTensorListInputData(TensorList *dst_tensorlist, TensorList *src_tensorlist) {
  MS_ASSERT(src_tensorlist != nullptr);
  MS_ASSERT(dst_tensorlist != nullptr);
  dst_tensorlist->FreeData();
  dst_tensorlist->ResetRefCount();
  dst_tensorlist->set_allocator(src_tensorlist->allocator());

  auto src_tensorlist_tensors_size = src_tensorlist->tensors().size();
  auto dst_tensorlist_tensors_size = dst_tensorlist->tensors().size();
  if (src_tensorlist_tensors_size != dst_tensorlist_tensors_size) {
    MS_LOG(ERROR) << "src tensorlist: " << src_tensorlist->tensor_name()
                  << " tesnors size: " << src_tensorlist_tensors_size
                  << " vs dst tensorlist: " << src_tensorlist->tensor_name()
                  << " tensors size: " << dst_tensorlist_tensors_size;
    return;
  }

  dst_tensorlist->set_own_data(src_tensorlist->own_data());
  for (size_t i = 0; i < src_tensorlist_tensors_size; ++i) {
    auto &src_tensor = src_tensorlist->tensors()[i];
    auto &dst_tensor = dst_tensorlist->tensors()[i];

    if (src_tensor->allocator() != nullptr) {
      src_tensor->allocator()->IncRefCount(src_tensor->data(), dst_tensor->ref_count());
    }
    dst_tensor->set_own_data(src_tensor->own_data());
    if (src_tensor->data() != nullptr) {
      dst_tensor->set_data(src_tensor->MutableData()); /* using MutableData to sync GPU data */
    }
    dst_tensor->set_shape(src_tensor->shape());
  }

  if (src_tensorlist->IsConst() || src_tensorlist->IsGraphInput()) {
    dst_tensorlist->set_own_data(false);
  } else {
    src_tensorlist->DecRefCount();
  }
}

int LiteOpActor::CastTensorListInputData(TensorList *dst_tensorlist, TensorList *src_tensorlist) {
  MS_ASSERT(src_tensorlist != nullptr);
  MS_ASSERT(dst_tensorlist != nullptr);
  dst_tensorlist->set_shape(src_tensorlist->shape());
  std::vector<std::vector<int>> tensors_shapes{};
  tensors_shapes.resize(src_tensorlist->tensors().size());
  for (size_t i = 0; i < tensors_shapes.size(); ++i) {
    tensors_shapes[i] = src_tensorlist->tensors()[i]->shape();
  }
  if (src_tensorlist->tensors_data_type() == kNumberTypeFloat16) {
    dst_tensorlist->MallocTensorListData(kNumberTypeFloat32, tensors_shapes);
  }
  if (src_tensorlist->tensors_data_type() == kNumberTypeFloat32) {
    dst_tensorlist->MallocTensorListData(kNumberTypeFloat16, tensors_shapes);
  }
  dst_tensorlist->set_allocator(src_tensorlist->allocator());
  dst_tensorlist->ResetRefCount();

  for (size_t i = 0; i < src_tensorlist->tensors().size(); ++i) {
    auto &src_tensor = src_tensorlist->tensors()[i];
    auto &dst_tensor = dst_tensorlist->tensors()[i];
    CastTensorInputData(dst_tensor, src_tensor);
  }
  return RET_OK;
}

int LiteSwitchOpActor::CompileTrueBranchArrow() {
  if (true_partial_node_ == nullptr) {
    MS_LOG(ERROR) << "true_partial_node_ is nullptr.";
    return RET_NULL_PTR;
  }
  auto subgraph = static_cast<kernel::PartialFusionKernel *>(true_partial_node_->kernel())->subgraph_kernel();
  auto true_branch_actor_id = subgraph_to_actor_.at(subgraph);

  for (size_t i = 0; i < true_partial_node_->in_tensors().size(); ++i) {
    int out_tensor_size = static_cast<int>(kernel_->out_tensors().size());
    for (int j = 0; j < out_tensor_size; ++j) {
      if (true_partial_node_->in_tensors()[i] != kernel_->out_tensors()[j]) {
        continue;
      }
      auto arrow = std::make_shared<DataArrow>(j, true_branch_actor_id, i);
      if (arrow == nullptr) {
        MS_LOG(ERROR) << "create DataArrow failed";
        return RET_ERROR;
      }
      true_branch_output_data_arrows_.emplace_back(std::move(arrow));
    }
  }
  return RET_OK;
}

int LiteSwitchOpActor::CompileFalseBranchArrow() {
  if (false_partial_node_ == nullptr) {
    MS_LOG(ERROR) << "false_partial_node_ is nullptr.";
    return RET_NULL_PTR;
  }
  auto subgraph = static_cast<kernel::PartialFusionKernel *>(false_partial_node_->kernel())->subgraph_kernel();
  auto false_branch_actor_id = subgraph_to_actor_.at(subgraph);

  for (size_t i = 0; i < false_partial_node_->in_tensors().size(); ++i) {
    int out_tensor_size = static_cast<int>(kernel_->out_tensors().size());
    for (int j = 0; j < out_tensor_size; ++j) {
      if (false_partial_node_->in_tensors()[i] != kernel_->out_tensors()[j]) {
        continue;
      }
      auto arrow = std::make_shared<DataArrow>(j, false_branch_actor_id, i);
      if (arrow == nullptr) {
        MS_LOG(ERROR) << "create DataArrow failed";
        return RET_ERROR;
      }
      false_branch_output_data_arrows_.emplace_back(std::move(arrow));
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

    if (switch_node->in_tensors().size() < kSwitchMinInputTensorSize) {
      MS_LOG(ERROR) << "actor name: " << this->GetAID() << "'s switch node " << switch_node->name()
                    << " input tensor size: " << switch_node->in_tensors().size() << " is less than 3.";
      return RET_ERROR;
    }

    switch_node_ = switch_node;
    if (switch_node->in_kernels().size() == kSwitchMaxInputKernelSize) {
      true_partial_node_ = switch_node->in_kernels().at(kSwitchTruePartialInputIndex);
      false_partial_node_ = switch_node->in_kernels().at(kSwitchFalsePartialInputIndex);
    }

    if (switch_node->in_kernels().size() == kSwitchMinInputKernelSize) {
      true_partial_node_ = switch_node->in_kernels().at(kSwitchTruePartialInputIndex - 1);
      false_partial_node_ = switch_node->in_kernels().at(kSwitchFalsePartialInputIndex - 1);
    }
    break;
  }
  return RET_OK;
}

void LiteSwitchOpActor::AppendOutputTensors() {
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
    true_branch_output_data_arrows_.clear();
    return ret;
  }

  ret = CompileFalseBranchArrow();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CompileFalseBranchArrow failed.";
    false_branch_output_data_arrows_.clear();
    true_branch_output_data_arrows_.clear();
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
    true_branch_output_data_arrows_.clear();
    false_branch_output_data_arrows_.clear();
    MS_LOG(ERROR) << "CompileArrowThroughSwitchCall failed.";
    return ret;
  }
  if (!true_branch_output_data_arrows_.empty() && !false_branch_output_data_arrows_.empty()) {
    MS_LOG(INFO) << "CompileArrowThroughSwitchCall done.";
    return RET_OK;
  }
  ret = CompileArrowThroughOutputKernels();
  if (ret != RET_OK) {
    output_data_arrows_.clear();
    MS_LOG(ERROR) << "CompileArrowThroughOutputKernels failed.";
    return ret;
  }
  return ret;
}

int LiteSwitchOpActor::PrepareOutputData() {
  true_branch_outputs_data_.resize(true_branch_output_data_arrows_.size());
  for (size_t i = 0; i < true_branch_output_data_arrows_.size(); i++) {
    auto &arrow = true_branch_output_data_arrows_[i];
    auto data =
      std::make_shared<OpData<Tensor>>(arrow->to_op_id_, (kernel_->out_tensors()).at(arrow->from_output_index_),
                                       static_cast<int>(arrow->to_input_index_));
    if (data == nullptr) {
      MS_LOG(ERROR) << "new true_branch_output_data failed.";
      return RET_NULL_PTR;
    }
    true_branch_outputs_data_.at(i) = data;
  }

  false_branch_outputs_data_.resize(false_branch_output_data_arrows_.size());
  for (size_t i = 0; i < false_branch_output_data_arrows_.size(); i++) {
    auto &arrow = false_branch_output_data_arrows_[i];
    auto data =
      std::make_shared<OpData<Tensor>>(arrow->to_op_id_, (kernel_->out_tensors()).at(arrow->from_output_index_),
                                       static_cast<int>(arrow->to_input_index_));
    if (data == nullptr) {
      MS_LOG(ERROR) << "new false_branch_output_data failed.";
      return RET_NULL_PTR;
    }
    false_branch_outputs_data_.at(i) = data;
  }
  return RET_OK;
}

void LiteSwitchOpActor::DecreaseTrueBranchInputTensor() {
  switch_node_->in_tensors()[kSwitchCondTensorIndex]->DecRefCount();
  for (auto input : true_partial_node_->in_tensors()) {
    input->DecRefCount();
  }
}

void LiteSwitchOpActor::DecreaseFalseBranchInputTensor() {
  switch_node_->in_tensors()[kSwitchCondTensorIndex]->DecRefCount();
  for (auto input : false_partial_node_->in_tensors()) {
    input->DecRefCount();
  }
}

void LiteSwitchOpActor::AsyncTrueBranchOutput(OpContext<Tensor> *context) {
  MS_ASSERT(true_branch_output_data_arrows_.size() == true_branch_outputs_data_.size());
  for (size_t i = 0; i < true_branch_output_data_arrows_.size(); ++i) {
    auto &data = true_branch_outputs_data_.at(i);
    Async(true_branch_output_data_arrows_[i]->to_op_id_, &mindspore::OpActor<Tensor>::RunOpData, data.get(), context);
  }
}

void LiteSwitchOpActor::AsyncFalseBranchOutput(OpContext<Tensor> *context) {
  MS_ASSERT(false_branch_output_data_arrows_.size() == false_branch_outputs_data_.size());
  for (size_t i = 0; i < false_branch_output_data_arrows_.size(); ++i) {
    auto &data = false_branch_outputs_data_.at(i);
    Async(false_branch_output_data_arrows_[i]->to_op_id_, &mindspore::OpActor<Tensor>::RunOpData, data.get(), context);
  }
}

void LiteSwitchOpActor::RunOpData(OpData<Tensor> *inputs, OpContext<Tensor> *context) {
  auto op_uuid = context->sequential_num_;
  input_op_datas_[op_uuid].push_back(inputs);
  inputs_data_[inputs->index_] = inputs->data_;
  if (input_op_datas_[op_uuid].size() < kernel_->in_tensors().size()) {
    return;
  }

  int ret = InitInputData();
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

  auto cond_ptr = reinterpret_cast<bool *>(switch_node_->in_tensors()[kSwitchCondTensorIndex]->data());
  if (cond_ptr == nullptr) {
    MS_LOG(ERROR) << "switch cond input data is nullptr.";
    context->SetFailed(RET_NULL_PTR);
    return;
  }
  if (*cond_ptr) {
    DecreaseFalseBranchInputTensor();
    AsyncTrueBranchOutput(context);
  } else {
    DecreaseTrueBranchInputTensor();
    AsyncFalseBranchOutput(context);
  }
}

#endif

void LiteOpActor::SetInputShape() {
  for (size_t i = 0; i < inputs_data_.size(); ++i) {
    auto &input_tensor = kernel_->in_tensors()[i];
    if (input_tensor->shape() == inputs_data_[i]->shape()) {
      continue;
    }
    MS_LOG(DEBUG) << "inputs_data_[" << i << "].shape: " << inputs_data_[i]->shape() << " vs kernel_->in_tensors()["
                  << i << "].shape: " << kernel_->in_tensors()[i]->shape() << " are not equal.";
    MS_LOG(DEBUG) << "this->kernel_->name(): " << this->kernel_->name();

    if (input_tensor->data_type() == kObjectTypeTensorType) {
#ifndef CONTROLFLOW_TENSORLIST_CLIP
      auto input_tensorlist = reinterpret_cast<TensorList *>(input_tensor);
      auto input_data_tensorlist = reinterpret_cast<TensorList *>(inputs_data_[i]);
      input_tensorlist->FreeTensorListData();
      input_tensorlist->set_element_shape(input_data_tensorlist->element_shape());
      input_tensorlist->set_shape(input_data_tensorlist->shape());
      std::vector<std::vector<int>> tensor_shape{};
      std::transform(input_data_tensorlist->tensors().begin(), input_data_tensorlist->tensors().end(),
                     std::back_inserter(tensor_shape), [](Tensor *tensor_item) { return tensor_item->shape(); });
      input_tensorlist->MallocTensorListData(input_data_tensorlist->tensors_data_type(), tensor_shape);
#endif
    } else {
      input_tensor->set_shape(inputs_data_[i]->shape());
      input_tensor->set_format(inputs_data_[i]->format());
    }
  }
}

int LiteOpActor::InitInputData() {
  SetInputShape();

  for (size_t i = 0; i < inputs_data_.size(); ++i) {
    auto dst_tensor = kernel_->in_tensors()[i];
    auto src_tensor = inputs_data_[i];
    if (dst_tensor->init_ref_count() == 0) {
      src_tensor->DecRefCount();
      continue;
    }

    if (NeedCastData(dst_tensor, src_tensor)) {
      CastInputData(dst_tensor, src_tensor);
      continue;
    }

    /* same data-type  */
    if (src_tensor->allocator() == nullptr || src_tensor->IsGraphInput()) {
      // delegate graph kernel output tensor
      SetInputData(dst_tensor, src_tensor);
    } else {
      MoveInputData(dst_tensor, src_tensor);
    }
  }
  return RET_OK;
}

void LiteOpActor::AsyncOutput(OpContext<Tensor> *context) {
  for (size_t i = 0; i < output_data_arrows_.size(); i++) {
    auto data = outputs_data_.at(i);
    Async(output_data_arrows_[i]->to_op_id_, &mindspore::OpActor<Tensor>::RunOpData, data.get(), context);
  }
}

void LiteOpActor::AddResultIndex(size_t index) { results_index_.push_back(index); }

void LiteOpActor::SetOutputData(OpContext<Tensor> *context) {
  for (auto index : results_index_) {
    context->SetResult(index, RET_OK);
  }
}

int LiteOpActor::PrepareOutputData() {
  outputs_data_.resize(output_data_arrows_.size());
  for (size_t i = 0; i < output_data_arrows_.size(); i++) {
    auto &arrow = output_data_arrows_[i];
    auto data =
      std::make_shared<OpData<Tensor>>(arrow->to_op_id_, (kernel_->out_tensors()).at(arrow->from_output_index_),
                                       static_cast<int>(arrow->to_input_index_));
    if (data == nullptr) {
      MS_LOG(ERROR) << "new output_data failed.";
      return RET_NULL_PTR;
    }
    outputs_data_.at(i) = data;
  }
  return RET_OK;
}

std::vector<std::shared_ptr<LiteOpActor>> CreateOpActor(const std::vector<kernel::LiteKernel *> &kernels,
                                                        const lite::InnerContext *ctx) {
  std::vector<std::shared_ptr<LiteOpActor>> actors;
  std::unordered_map<kernel::LiteKernel *, AID> subgraph_name_AID_map{};
  ActorThreadPool *thread_pool = reinterpret_cast<ActorThreadPool *>(ctx->thread_pool());
  if (thread_pool == nullptr) {
    MS_LOG(ERROR) << "thread pool is nullptr";
    return actors;
  }
  for (auto &kernel : kernels) {
    /* make subgraph name (actor name) unique */
    kernel->set_name(kernel->name() + "_" + to_string(actor_count++));
#ifndef CONTROLFLOW_TENSORLIST_CLIP
    if ((kernel::LiteKernelUtil::IsSwitchCall(kernel))) {
      auto switch_actor = std::make_shared<LiteSwitchOpActor>(kernel);
      if (switch_actor == nullptr) {
        MS_LOG(ERROR) << "create LiteSwitchOpActor failed: " << kernel->name();
        actors.clear();
        return actors;
      }
      switch_actor->set_thread_pool(thread_pool);
      subgraph_name_AID_map[kernel] = switch_actor->GetAID();
      actors.push_back(switch_actor);
    } else {
#endif
      auto actor = std::make_shared<LiteOpActor>(kernel);
      if (actor == nullptr) {
        MS_LOG(ERROR) << "create LiteOpActor failed: " << kernel->name();
        actors.clear();
        return actors;
      }
      actor->set_thread_pool(thread_pool);
      subgraph_name_AID_map[kernel] = actor->GetAID();
      actors.push_back(actor);
#ifndef CONTROLFLOW_TENSORLIST_CLIP
    }
#endif
  }

  for (auto &actor : actors) {
    actor->SetSubgraphAIDMap(subgraph_name_AID_map);
    auto aid = mindspore::Spawn(actor);
  }
  return actors;
}

int MindrtInit() { return mindspore::Initialize("", "", "", ""); }

void MindrtTerminate(const std::vector<std::shared_ptr<LiteOpActor>> &actor_list) {
  for (const auto &actor : actor_list) {
    mindspore::Terminate(actor->GetAID());
  }
}
}  // namespace mindspore::lite
