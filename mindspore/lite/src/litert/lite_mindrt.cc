/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "src/litert/lite_mindrt.h"
#include "mindrt/include/mindrt.hpp"
#include "src/litert/kernel_exec_util.h"
#include "src/common/tensor_util.h"
#include "src/common/common.h"
#include "src/litert/inner_allocator.h"
#include "src/litert/kernel/cpu/base/partial_fusion.h"
#include "src/control_flow/control_actor_creator.h"

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
    MS_LOG(ERROR) << "run kernel failed, name: " << kernel_->name();
    context->SetFailed(ret);
    return;
  }

  ret = kernel_->Execute(*(reinterpret_cast<const KernelCallBack *>(context->kernel_call_back_before_)),
                         *(reinterpret_cast<const KernelCallBack *>(context->kernel_call_back_after_)));
  input_op_datas_.erase(op_uuid);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "run kernel failed, name: " << kernel_->name();
    context->SetFailed(ret);
    return;
  }
  AsyncOutput(context);
  SetOutputData(context);
  return;
}

bool OfflineIsolated(const std::vector<kernel::KernelExec *> &kernels, const kernel::KernelExec &this_kernel,
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

TypeId GetSubgraphInTensorDataType(const kernel::KernelExec *kernel, const lite::Tensor *tensor) {
#ifdef ENABLE_LITE_ACL
  if (kernel->subgraph_type() == kernel::kCustomSubGraph) {
    return tensor->data_type();
  }
#endif
  if (kernel->subgraph_type() != kernel::kGpuFp16SubGraph || tensor->IsGraphInput() || tensor->IsGraphOutput()) {
    if (tensor->data_type() == kNumberTypeFloat16 || tensor->data_type() == kNumberTypeFloat32) {
      return kernel->desc().data_type;
    }
  }
  return tensor->data_type();
}

int LiteOpActor::PreInit(std::vector<std::shared_ptr<LiteOpActor>> *actors,
                         std::unordered_map<Tensor *, Tensor *> *input_map) {
  return IsolateInputData(actors, input_map);
}
int LiteOpActor::PostInit() { return PrepareOutputData(); }

int LiteOpActor::IsolateInputData(std::vector<std::shared_ptr<LiteOpActor>> *actors,
                                  std::unordered_map<Tensor *, Tensor *> *input_map) {
  isolate_input_map_ = input_map;
  std::vector<kernel::KernelExec *> kernels{};
  std::transform(actors->begin(), actors->end(), std::back_inserter(kernels),
                 [](const std::shared_ptr<LiteOpActor> &actor) { return actor->kernel_; });
  size_t in_tensor_size = kernel_->in_tensors().size();
  for (size_t i = 0; i < in_tensor_size; i++) {
    Tensor *old_tensor = kernel_->in_tensors()[i];

    if (OfflineIsolated(kernels, *kernel_, *old_tensor)) {
      if (old_tensor->data_type() == kNumberTypeFloat16 || old_tensor->data_type() == kNumberTypeFloat32) {
        old_tensor->set_data_type(kernel_->desc().data_type);
      }
      SetTensorListTensorDataType(kernel_->desc().data_type, old_tensor);
      old_tensor->set_allocator(kernel_->Context()->allocator);
      continue;
    }

    TypeId new_data_type = GetSubgraphInTensorDataType(kernel_, old_tensor);
    Tensor *new_tensor =
      new (std::nothrow) Tensor(new_data_type, old_tensor->shape(), old_tensor->format(), old_tensor->category());
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
    auto ret = kernel::KernelExecUtil::ReplaceSubGraphNodesInTensor(kernel_, old_tensor, new_tensor);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ReplaceSubGraphNodesInTensor failed.";
      return ret;
    }

    // for case that subgraph input is subgraph output, replace old_tensor with new_tensor
    ctx_->ReplaceLinkInfoSenderWithNewOne(new_tensor, old_tensor);

    // keep new link info for isolate input data case.
    ctx_->SetLinkInfo(old_tensor, new_tensor);

    /* set subgraph input for copy data */
    kernel_->set_in_tensor(new_tensor, i);
  }

  for (auto &item : *isolate_input_map_) {
    ctx_->ReplaceLinkInfoReceiverWithNewOne(item.first, item.second);
  }

  return RET_OK;
}

int LiteOpActor::ResizeGraphInput(const std::vector<mindspore::lite::Tensor *> &inputs,
                                  const std::vector<std::vector<int>> &dims) {
  for (auto map : *isolate_input_map_) {
    auto isolate_tensor = map.first;
    auto src_tensor = map.second;
    for (size_t i = 0; i < inputs.size(); i++) {
      if (src_tensor == inputs[i]) {
        isolate_tensor->FreeData();
        isolate_tensor->set_shape(dims[i]);
      }
    }
  }
  return RET_OK;
}

int LiteOpActor::CompileArrow(const std::unordered_map<void *, std::set<std::pair<AID, size_t>>> &receivers_map) {
  auto ret = UpdateActorOutput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "update actor output failed.";
    return ret;
  }

  return CompileArrowThroughOutputTensors(receivers_map);
}

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
  std::vector<kernel::KernelExec *> call_kernels{};
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

  auto partial_nodes = kernel::KernelExecUtil::GetCallInputPartials(call_node_);
  if (partial_nodes.size() != 1) {
    MS_LOG(ERROR) << "partial output is not right.";
    return RET_ERROR;
  }
  partial_node_ = partial_nodes.front();
  (void)std::copy(partial_node_->in_tensors().begin(), partial_node_->in_tensors().end(),
                  std::back_inserter(origin_output_tensors));

  kernel_->set_out_tensors(origin_output_tensors);

  subgraph_kernel->DropNode(partial_node_);
  subgraph_kernel->DropNode(call_node_);
  return RET_OK;
}

bool LiteOpActor::ArrowHasCompiled(const AID &actor_name, size_t to_index,
                                   const std::unordered_map<AID, std::set<size_t>> &receiver_index_set) {
  auto iter = receiver_index_set.find(actor_name);
  if (iter != receiver_index_set.end()) {
    return iter->second.find(to_index) != iter->second.end();
  }
  return false;
}

void LiteOpActor::MarkArrowAsCompiled(const AID *actor_name, size_t to_index,
                                      std::unordered_map<AID, std::set<size_t>> *receiver_index_set) {
  if (receiver_index_set->find(*actor_name) == receiver_index_set->end()) {
    std::set<size_t> tmp{to_index};
    receiver_index_set->insert(std::pair<AID, std::set<size_t>>(*actor_name, tmp));
  } else {
    (void)receiver_index_set->at(*actor_name).insert(to_index);
  }
}

int LiteOpActor::CreateCommonArrow(const std::unordered_map<void *, std::set<std::pair<AID, size_t>>> &receivers_map,
                                   const std::set<void *> &receiver_tensors, const size_t &output_index,
                                   std::unordered_map<AID, std::set<size_t>> *receiver_index_set) {
  std::unordered_map<void *, std::set<std::pair<AID, size_t>>>::const_iterator iter;
  for (auto receiver_tensor : receiver_tensors) {
    iter = receivers_map.find(receiver_tensor);
    if (iter == receivers_map.end()) {
      MS_LOG(DEBUG) << "not a useful receiver.";
      continue;
    }
    auto receiver_set = iter->second;
    for (auto item : receiver_set) {
      if (ArrowHasCompiled(item.first, item.second, *receiver_index_set)) {
        continue;
      }
      MarkArrowAsCompiled(&(item.first), item.second, receiver_index_set);
      auto arrow = std::make_shared<DataArrow>(output_index, item.first, item.second);
      MS_CHECK_TRUE_MSG(arrow != nullptr, RET_ERROR, "create arrow failed.");
      output_data_arrows_.push_back(arrow);
    }
  }
  return RET_OK;
}

int LiteOpActor::CreateEmptyArrow(const size_t &output_index) {
  AID non;
  auto arrow = std::make_shared<DataArrow>(output_index, non, output_index);
  MS_CHECK_TRUE_MSG(arrow != nullptr, RET_ERROR, "create arrow failed.");
  output_data_arrows_.push_back(arrow);
  return RET_OK;
}

int LiteOpActor::CompileArrowThroughOutputTensors(
  const std::unordered_map<void *, std::set<std::pair<AID, size_t>>> &receivers_map) {
  auto output_tensors = this->kernel_->out_tensors();
  auto output_tensors_size = output_tensors.size();

  std::unordered_map<AID, std::set<size_t>> receiver_index_set{};
  for (size_t i = 0; i < output_tensors_size; ++i) {
    auto receiver_tensors = ctx_->GetLinkInfo(output_tensors[i]);
    if (receiver_tensors.empty()) {
      MS_LOG(DEBUG) << "create when running.";
      auto ret = CreateEmptyArrow(i);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "CreateEmptyArrow failed, output tensor name: " << output_tensors[i]->tensor_name();
        return ret;
      }
      continue;
    }
    auto ret = CreateCommonArrow(receivers_map, receiver_tensors, i, &receiver_index_set);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "CreateCommonArrow failed, output tensor name: " << output_tensors[i]->tensor_name();
      return ret;
    }
  }
  return RET_OK;
}

int LiteOpActor::SetInputShape() {
  auto ret = RET_OK;
  for (size_t i = 0; i < inputs_data_.size(); ++i) {
    auto &input_tensor = kernel_->in_tensors()[i];
    if (input_tensor->shape() == inputs_data_[i]->shape()) {
      continue;
    }
    ret = SetTensorShape(input_tensor, inputs_data_[i]);
    MS_CHECK_FALSE_MSG(ret != RET_OK, ret, "set input shape failed.");
  }
  return RET_OK;
}

int LiteOpActor::AssignInputData() {
  auto ret = RET_OK;
  for (size_t i = 0; i < inputs_data_.size(); ++i) {
    auto dst_tensor = kernel_->in_tensors()[i];
    auto src_tensor = inputs_data_[i];
    dst_tensor->set_shape_changed(src_tensor->get_shape_changed());
    if (dst_tensor->init_ref_count() == 0) {
      src_tensor->DecRefCount();
      continue;
    }
    if (NeedCastData(dst_tensor, src_tensor)) {
      ret = CastTensorData(dst_tensor, src_tensor, support_fp16_);
      MS_CHECK_FALSE_MSG(ret != RET_OK, ret, "CastTensorData failed.");
      continue;
    }
    /* same data-type  */
    if (src_tensor->allocator() == nullptr || src_tensor->IsGraphInput()) {
      // delegate graph kernel output tensor
      ret = SetTensorData(dst_tensor, src_tensor);
      MS_CHECK_FALSE_MSG(ret != RET_OK, ret, "SetTensorData failed.");
    } else {
      ret = MoveTensorData(dst_tensor, src_tensor);
      MS_CHECK_FALSE_MSG(ret != RET_OK, ret, "MoveTensorData failed.");
    }
  }
  return ret;
}

bool LiteOpActor::NeedResize() {
  for (size_t i = 0; i < inputs_data_.size(); ++i) {
    auto &subgraph_input = kernel_->in_tensors()[i];
    auto &cur_input = inputs_data_[i];
    if (!IsSameShape(subgraph_input, cur_input)) {
      return true;
    }
  }
  return false;
}

int LiteOpActor::InitInputData() {
  for (size_t i = 0; i < inputs_data_.size(); ++i) {
    if (inputs_data_[i] == nullptr) {
      MS_LOG(ERROR) << "inputs_data_ nullptr, index: " << i;
      return RET_ERROR;
    }
  }
  bool need_resize = NeedResize();
  auto ret = SetInputShape();
  MS_CHECK_FALSE_MSG(ret != RET_OK, ret, "Set input shape failed.");
  if (need_resize) {
    auto subgraph_kernel = reinterpret_cast<kernel::SubGraphKernel *>(kernel_);
    MS_CHECK_FALSE_MSG(subgraph_kernel == nullptr, RET_ERROR, "Lite actor, cast kernel to subgraph kernel failed.");
    ret = subgraph_kernel->MallocSubgraphInputs();
    MS_CHECK_FALSE_MSG(ret != RET_OK, ret, "Subgraph kernel MallocSubgraphInputs failed.");
  }
  ret = AssignInputData();
  MS_CHECK_FALSE_MSG(ret != RET_OK, ret, "Subgraph kernel AssignInputData failed.");
  if (need_resize) {
    auto subgraph_kernel = reinterpret_cast<kernel::SubGraphKernel *>(kernel_);
    ret = subgraph_kernel->ReSize();
    MS_CHECK_FALSE_MSG(ret != RET_OK, ret, "Subgraph kernel Resize failed.");
    ret = subgraph_kernel->MallocNodesOutputSpace();
    MS_CHECK_FALSE_MSG(ret != RET_OK, ret, "Subgraph kernel MallocSubgraphInputs failed.");
  }
  return RET_OK;
}

void LiteOpActor::AsyncOutput(OpContext<Tensor> *context) {
  auto output_size = output_data_arrows_.size();
  for (size_t i = 0; i < output_size; ++i) {
    auto data = outputs_data_[i];
    Async(output_data_arrows_[i]->to_op_id_, get_actor_mgr(), &mindspore::OpActor<Tensor>::RunOpData, data.get(),
          context);
  }
}

void LiteOpActor::AddResultIndex(size_t index, size_t tensor_index) {
  results_index_.push_back(index);
  results_tensor_index_.push_back(tensor_index);
}

void LiteOpActor::SetOutputData(const OpContext<Tensor> *context) {
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
    if (MS_UNLIKELY(data == nullptr)) {
      MS_LOG(ERROR) << "new output_data failed.";
      return RET_NULL_PTR;
    }
    outputs_data_[i] = data;
  }
  return RET_OK;
}

std::vector<std::shared_ptr<LiteOpActor>> CreateOpActor(const std::vector<kernel::KernelExec *> &kernels,
                                                        lite::InnerContext *ctx, const std::shared_ptr<ActorMgr> &mgr) {
  std::vector<std::shared_ptr<LiteOpActor>> actors;
  ActorThreadPool *thread_pool = reinterpret_cast<ActorThreadPool *>(ctx->thread_pool_);
  if (thread_pool == nullptr) {
    MS_LOG(ERROR) << "thread pool is nullptr";
    return actors;
  }
  actors.reserve(kernels.size());
  for (auto &kernel : kernels) {
    /* make subgraph name (actor name) unique */
    kernel->set_name(kernel->name() + "_" + std::to_string(actor_count++));
    std::shared_ptr<LiteOpActor> actor = CreateActor(kernel, ctx);
    if (actor == nullptr) {
      MS_LOG(ERROR) << "create LiteOpActor failed: " << kernel->name();
      actors.clear();
      return actors;
    }
    actor->set_thread_pool(thread_pool);
    actor->set_actor_mgr(mgr);
    actors.push_back(actor);
  }

  for (auto &actor : actors) {
    auto aid = mindspore::Spawn(actor);
  }
  return actors;
}

int MindrtInit() { return mindspore::Initialize("", "", "", ""); }

void MindrtTerminate(const std::vector<std::shared_ptr<LiteOpActor>> &actor_list,
                     const std::shared_ptr<ActorMgr> &actor_mgr) {
  for (const auto &actor : actor_list) {
    mindspore::Terminate(actor->GetAID(), actor_mgr);
  }
}
}  // namespace mindspore::lite
