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
#ifndef CONTROLFLOW_TENSORLIST_CLIP
#include "src/control_flow/actor/switch_actor.h"
#include "src/control_flow/actor/entrance_actor.h"
#include "src/control_flow/actor/exit_actor.h"
#endif

namespace mindspore::lite {
void LiteOpActor::RunOpData(OpData<lite::Tensor> *inputs, OpContext<lite::Tensor> *context) {
  auto op_uuid = context->sequential_num_;
  input_op_datas_[op_uuid].push_back(inputs);
  inputs_data_[inputs->index_] = inputs->data_;
  if (input_op_datas_[op_uuid].size() < kernel_->in_tensors().size()) {
    return;
  }

  InitInputData();

  auto ret = RunKernel(*(reinterpret_cast<const KernelCallBack *>(context->kernel_call_back_before_)),
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
                    [&this_input_tensor](const lite::Tensor *tensor) { return tensor == &this_input_tensor; })) {
      return false;
    }
  }
  return true;
}

int LiteOpActor::PreInit(std::vector<std::shared_ptr<LiteOpActor>> *actors,
                         std::unordered_map<Tensor *, Tensor *> *input_map) {
  return IsolateInputData(actors, input_map);
}
int LiteOpActor::PostInit() { return PrepareOutputData(); }

int LiteOpActor::IsolateInputData(std::vector<std::shared_ptr<LiteOpActor>> *actors,
                                  std::unordered_map<Tensor *, Tensor *> *input_map) {
  isolate_input_map_ = input_map;
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
    isolate_input_map_->insert(std::make_pair(new_tensor, old_tensor));
    auto ret = kernel::LiteKernelUtil::ReplaceSubGraphNodesInTensor(kernel_, old_tensor, new_tensor);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ReplaceSubGraphNodesInTensor failed.";
      return ret;
    }

    // for case that subgraph input is subgraph output, replace old_tensor with new_tensor
    auto receiver_set = ctx_->GetLinkInfo(old_tensor);
    for (auto item : receiver_set) {
      ctx_->SetLinkInfo(new_tensor, item);
    }

    // keep new link info for isolate input data case.
    ctx_->SetLinkInfo(old_tensor, new_tensor);

    /* set subgraph input for copy data */
    kernel_->set_in_tensor(new_tensor, i);
  }
  return RET_OK;
}

int LiteOpActor::ResizeGraphInput(const std::vector<mindspore::tensor::MSTensor *> &inputs,
                                  const std::vector<std::vector<int>> &dims) {
  for (auto map : *isolate_input_map_) {
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

int LiteOpActor::CompileArrow(const std::unordered_map<void *, std::set<std::pair<AID, size_t>>> &receivers_map) {
#ifndef CONTROLFLOW_TENSORLIST_CLIP
  auto ret = UpdateActorOutput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "update actor output failed.";
    return ret;
  }
#endif

  return CompileArrowThroughOutputTensors(receivers_map);
}

#ifndef CONTROLFLOW_TENSORLIST_CLIP
int LiteOpActor::UpdateActorOutput() {
  if (kernel_->desc().arch == kernel::kDelegate) {
    MS_LOG(DEBUG) << "no need for delegate kernel.";
    return RET_OK;
  }
  auto *subgraph_kernel = reinterpret_cast<kernel::SubGraphKernel *>(kernel_);
  if (subgraph_kernel == nullptr) {
    MS_LOG(INFO) << "kernel is not subgraph kernel, no partial call.";
    return RET_OK;
  }
  auto output_kernels = subgraph_kernel->out_nodes();
  std::vector<kernel::LiteKernel *> call_kernels{};
  for (auto output_kernel : output_kernels) {
    if (output_kernel->type() == schema::PrimitiveType_Call) {
      call_kernels.push_back(output_kernel);
    }
  }
  if (call_kernels.empty()) {
    MS_LOG(DEBUG) << "not end with call kernel, no need to update output.";
    return RET_OK;
  }
  if (call_kernels.size() != 1) {
    MS_LOG(ERROR) << "not support many call kernels in one subgraph.";
    return RET_NOT_SUPPORT;
  }
  call_node_ = call_kernels.front();

  // erase call output tensor
  auto origin_output_tensors = kernel_->out_tensors();
  auto call_output_tensors = call_node_->out_tensors();

  for (auto iter = origin_output_tensors.begin(); iter != origin_output_tensors.end();) {
    if (IsContain(call_output_tensors, *iter)) {
      iter = origin_output_tensors.erase(iter);
    } else {
      ++iter;
    }
  }

  auto partial_nodes = kernel::LiteKernelUtil::GetCallInputPartials(call_node_);
  if (partial_nodes.size() != 1) {
    MS_LOG(ERROR) << "partial output is not right.";
    return RET_ERROR;
  }
  partial_node_ = partial_nodes.front();
  std::copy(partial_node_->in_tensors().begin(), partial_node_->in_tensors().end(),
            std::back_inserter(origin_output_tensors));

  kernel_->set_out_tensors(origin_output_tensors);

  subgraph_kernel->DropNode(partial_node_);
  subgraph_kernel->DropNode(call_node_);
  return RET_OK;
}
#endif

bool LiteOpActor::ArrowHasCompiled(const AID &actor_name, const size_t &to_index,
                                   const std::unordered_map<AID, std::set<size_t>> &receiver_index_set) {
  if (receiver_index_set.find(actor_name) != receiver_index_set.end()) {
    return receiver_index_set.at(actor_name).find(to_index) != receiver_index_set.at(actor_name).end();
  }
  return false;
}

void LiteOpActor::MarkArrowAsCompiled(const AID *actor_name, const size_t *to_index,
                                      std::unordered_map<AID, std::set<size_t>> *receiver_index_set) {
  if (receiver_index_set->find(*actor_name) == receiver_index_set->end()) {
    std::set<size_t> tmp{*to_index};
    receiver_index_set->insert(std::pair<AID, std::set<size_t>>(*actor_name, tmp));
  } else {
    receiver_index_set->at(*actor_name).insert(*to_index);
  }
}

int LiteOpActor::CreateCommonArrow(const std::unordered_map<void *, std::set<std::pair<AID, size_t>>> &receivers_map,
                                   const std::set<void *> &subgraph_inputs_set,
                                   const std::set<void *> &receiver_tensors, const size_t &output_index,
                                   std::unordered_map<AID, std::set<size_t>> *receiver_index_set) {
  for (auto receiver_tensor : receiver_tensors) {
    MS_CHECK_TRUE_MSG(receivers_map.find(receiver_tensor) != receivers_map.end(), RET_ERROR,
                      "not find receiver_tensor in receivers_map");
    if (partial_node_ != nullptr && subgraph_inputs_set.find(receiver_tensor) == subgraph_inputs_set.end()) {
      MS_LOG(DEBUG) << "not a need arrow for this actor.";
      continue;
    }
    auto receiver_set = receivers_map.at(receiver_tensor);
    for (auto item : receiver_set) {
      if (ArrowHasCompiled(item.first, item.second, *receiver_index_set)) {
        continue;
      }
      MarkArrowAsCompiled(&(item.first), &(item.second), receiver_index_set);
      auto arrow = std::make_shared<DataArrow>(output_index, item.first, item.second);
      MS_CHECK_TRUE_MSG(arrow != nullptr, RET_ERROR, "create arrow failed.");
      output_data_arrows_.push_back(arrow);
    }
  }

  return RET_OK;
}

int LiteOpActor::CompileArrowThroughOutputTensors(
  const std::unordered_map<void *, std::set<std::pair<AID, size_t>>> &receivers_map) {
  auto output_tensors = this->kernel_->out_tensors();
  auto output_tensors_size = output_tensors.size();
  auto subgraph_inputs_set = PartialSubgraphInputTensors(partial_node_);

  std::unordered_map<AID, std::set<size_t>> receiver_index_set{};
  for (size_t i = 0; i < output_tensors_size; ++i) {
    auto receiver_tensors = ctx_->GetLinkInfo(output_tensors[i]);
    if (receiver_tensors.empty()) {
      MS_LOG(DEBUG) << "may be graph output.";
      continue;
    }
    auto ret = CreateCommonArrow(receivers_map, subgraph_inputs_set, receiver_tensors, i, &receiver_index_set);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "CreateCommonArrow failed, output tensor name: " << output_tensors[i]->tensor_name();
      return ret;
    }
  }
  return RET_OK;
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
#endif

std::set<void *> LiteOpActor::PartialSubgraphInputTensors(kernel::LiteKernel *partial_node) {
  if (partial_node == nullptr) {
    return {};
  }
  auto partial_kernel = reinterpret_cast<kernel::PartialFusionKernel *>(partial_node->kernel());
  if (partial_kernel == nullptr) {
    MS_LOG(WARNING) << "cast to partial kernel failed.";
    return {};
  }
  std::set<void *> ret{};
  auto partial_subgraph_kernels = partial_kernel->subgraph_kernels();
  if (partial_subgraph_kernels.empty()) {
    MS_LOG(ERROR) << "partial's subgraph kernel is empty.";
    return {};
  }
  // the first subgraph kernel is the input subgraph kernel
  auto subgraph = partial_subgraph_kernels.front();
  auto inputs = subgraph->in_tensors();
  for (auto &input : inputs) {
    ret.insert(input);
  }
  return ret;
}

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
                     std::back_inserter(tensor_shape), [](const Tensor *tensor_item) { return tensor_item->shape(); });
      input_tensorlist->MallocTensorListData(input_data_tensorlist->tensors_data_type(), tensor_shape);
#endif
    } else {
      input_tensor->set_shape(inputs_data_[i]->shape());
      input_tensor->set_format(inputs_data_[i]->format());
    }
  }
}

void LiteOpActor::InitInputData() {
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
  return;
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

std::vector<std::shared_ptr<LiteOpActor>> CreateOpActor(const std::vector<kernel::LiteKernel *> &kernels,
                                                        lite::InnerContext *ctx) {
  std::vector<std::shared_ptr<LiteOpActor>> actors;
  ActorThreadPool *thread_pool = reinterpret_cast<ActorThreadPool *>(ctx->thread_pool());
  if (thread_pool == nullptr) {
    MS_LOG(ERROR) << "thread pool is nullptr";
    return actors;
  }
  for (auto &kernel : kernels) {
    /* make subgraph name (actor name) unique */
    kernel->set_name(kernel->name() + "_" + to_string(actor_count++));
    std::shared_ptr<LiteOpActor> actor = nullptr;
#ifndef CONTROLFLOW_TENSORLIST_CLIP
    if ((kernel::LiteKernelUtil::IsSwitchTypeCall(kernel))) {
      actor = std::make_shared<LiteSwitchOpActor>(kernel, ctx);
    } else if (kernel->subgraph_type() == kernel::kEntranceSubGraph) {
      actor = std::make_shared<LiteEntranceOpActor>(kernel, ctx);
    } else if (kernel->subgraph_type() == kernel::kExitSubGraph) {
      actor = std::make_shared<LiteExitOpActor>(kernel, ctx);
    } else {
#endif
      actor = std::make_shared<LiteOpActor>(kernel, ctx);
#ifndef CONTROLFLOW_TENSORLIST_CLIP
    }
#endif
    if (actor == nullptr) {
      MS_LOG(ERROR) << "create LiteOpActor failed: " << kernel->name();
      actors.clear();
      return actors;
    }
    actor->set_thread_pool(thread_pool);
    actors.push_back(actor);
  }

  for (auto &actor : actors) {
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
