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

#include "backend/ms_backend/runtime/actors/control_flow/control_actor.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/parameter_store.h"
#include "primitive/framework_ops.h"
#include "utils/profile.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace runtime {
ControlActor::ControlActor(const std::string &name, KernelTransformType type, const AID &memory_manager_aid,
                           const std::vector<KernelWithIndex> &parameters, const AnfNodePtr &node)
    : MemoryAwareActor(name, type, nullptr, memory_manager_aid), formal_parameters_(parameters), node_(node) {
  input_partials_.resize(parameters.size());
  input_kernel_tensors_.resize(parameters.size());
  output_data_by_output_index_.resize(parameters.size());
}

void ControlActor::Init() {
  InitOutputData();
  if (output_data_.size() != output_data_arrows_.size()) {
    MS_LOG(EXCEPTION) << "The output data size is wrong: " << GetAID().Name();
  }

  size_t output_data_index = 0;
  for (size_t i = 0; i < output_data_arrows_.size(); ++i) {
    const auto &data_arrow = output_data_arrows_[i];
    auto data = output_data_[output_data_index].first.get();
    MS_EXCEPTION_IF_NULL(data);
    MS_EXCEPTION_IF_NULL(data_arrow);
    if (IntToSize(data_arrow->from_output_index_) >= output_data_by_output_index_.size()) {
      MS_LOG(EXCEPTION) << "The output index is out of range: " << GetAID();
    }
    if (i < output_need_disable_dynamic_ref_counts_.size() && output_need_disable_dynamic_ref_counts_[i] &&
        data->data_ != nullptr && data->data_->device_address() != nullptr) {
      data->data_->UpdateFlag(device::kDeviceAddressFlagNullptr);
      MS_LOG(INFO) << "Add null flag for device address:" << data->data_->device_address().get()
                   << " in actor:" << GetAID();
    }
    (void)output_data_by_output_index_[IntToSize(data_arrow->from_output_index_)].emplace_back(data);
    ++output_data_index;
  }
}

void ControlActor::GetAllKernelTensors(const OpPartialPtr &op_partial, std::vector<KernelTensorPtr> *kernel_tensors) {
  MS_EXCEPTION_IF_NULL(op_partial);
  (void)std::transform(op_partial->kernel_tensors_.begin(), op_partial->kernel_tensors_.end(),
                       std::back_inserter(*kernel_tensors),
                       [](const auto &kernel_tensor) { return kernel_tensor.second; });

  // Foreach the op partial to fetch the device tensors.
  for (auto &partial : op_partial->partials_) {
    GetAllKernelTensors(partial.second, kernel_tensors);
  }
}

void ControlActor::GetAllKernelTensors(const OpRealParameterWithBranchID &op_real_parameter,
                                       std::vector<KernelTensorPtr> *kernel_tensors) {
  MS_EXCEPTION_IF_NULL(kernel_tensors);
  for (auto &kernel_tensor : op_real_parameter.kernel_tensors_) {
    (void)kernel_tensors->emplace_back(kernel_tensor.second);
  }

  // Foreach the op partial to fetch the device tensors.
  for (auto &partial : op_real_parameter.partials_) {
    GetAllKernelTensors(partial.second, kernel_tensors);
  }
}

void ControlActor::IncreaseNewRefCountForPartial(const OpPartialPtr &op_partial) {
  if (op_partial == nullptr) {
    MS_LOG(EXCEPTION) << "Empty op partial for actor:" << GetAID();
  }
  std::vector<KernelTensorPtr> partial_kernel_tensors;
  GetAllKernelTensors(op_partial, &partial_kernel_tensors);
  for (auto &partial_kernel_tensor : partial_kernel_tensors) {
    MS_EXCEPTION_IF_NULL(partial_kernel_tensor);
    partial_kernel_tensor->IncreaseNewRefCount(GetAID().Name());
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Increase new ref count for kernel tensor:" << partial_kernel_tensor->ToString() << " in actor:" << GetAID();
  }
}

void ControlActor::IncreaseNewRefCountForRealParameter(const OpRealParameterWithBranchID &op_real_parameter) {
  std::vector<KernelTensorPtr> partial_kernel_tensors;
  GetAllKernelTensors(op_real_parameter, &partial_kernel_tensors);
  for (auto &partial_kernel_tensor : partial_kernel_tensors) {
    MS_EXCEPTION_IF_NULL(partial_kernel_tensor);
    partial_kernel_tensor->IncreaseNewRefCount(GetAID().Name());
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Increase new ref count for kernel tensor:" << partial_kernel_tensor->ToString() << " in actor:" << GetAID();
  }
}

size_t ControlActor::FetchNodePosition(const KernelWithIndex &node) const {
  const auto &iter = find(formal_parameters_.begin(), formal_parameters_.end(), node);
  if (iter == formal_parameters_.end()) {
    const auto &load_iter =
      std::find_if(formal_parameters_.begin(), formal_parameters_.end(), [&node](const KernelWithIndex &pair) {
        return pair.first != nullptr && common::AnfAlgo::CheckPrimitiveType(pair.first, prim::kPrimLoad) &&
               pair.first->cast<CNodePtr>()->input(1) == node.first && node.second == 0;
      });
    if (load_iter != formal_parameters_.end()) {
      return load_iter - formal_parameters_.begin();
    }
    const auto &get_item_iter =
      std::find_if(formal_parameters_.begin(), formal_parameters_.end(), [&node](const KernelWithIndex &pair) {
        return pair.first != nullptr &&
               ((common::AnfAlgo::CheckPrimitiveType(pair.first, prim::kPrimTupleGetItem) &&
                 common::AnfAlgo::GetTupleGetItemRealInput(pair.first->cast<CNodePtr>()) == node.first &&
                 common::AnfAlgo::GetTupleGetItemOutIndex(pair.first->cast<CNodePtr>()) == node.second) ||
                node == common::AnfAlgo::VisitKernelWithReturnType(pair.first, pair.second, false));
      });
    if (get_item_iter != formal_parameters_.end()) {
      MS_LOG(INFO) << "Input node:" << node.first->DebugString() << " fullname:" << node.first->fullname_with_scope()
                   << " node ptr:" << node.first << " index:" << node.second
                   << " match the tuple get item node:" << get_item_iter->first->DebugString()
                   << " fullname:" << get_item_iter->first->fullname_with_scope()
                   << " node ptr:" << get_item_iter->first << " index:" << get_item_iter->second;
      return get_item_iter - formal_parameters_.begin();
    }
    for (const auto &formal_parameter : formal_parameters_) {
      MS_LOG(WARNING) << "Actor:" << GetAID() << " formal parameter:"
                      << (formal_parameter.first != nullptr ? formal_parameter.first->DebugString() : "")
                      << " full name:"
                      << (formal_parameter.first != nullptr ? formal_parameter.first->fullname_with_scope() : "")
                      << " index:" << formal_parameter.second << " node ptr:" << formal_parameter.first;
    }
    MS_LOG_WITH_NODE(EXCEPTION, node.first)
      << "Invalid formal parameter:" << (node.first != nullptr ? node.first->DebugString() : "")
      << " full name:" << (node.first != nullptr ? node.first->fullname_with_scope() : "") << " node ptr:" << node.first
      << " index:" << node.second << " for actor:" << GetAID();
  }
  return iter - formal_parameters_.begin();
}

void ControlActor::Run(OpContext<KernelTensor> *const context) {
  try {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_ACTOR) << "Control actor:" << GetAID() << " start run.";
    // The exit actor is the output of kernel graph when the node_ is null.
    if (type_ == KernelTransformType::kExitActor && node_ == nullptr) {
      double end_time = GetTime();
      const size_t kSecondsToMilliseconds = 1000;
      MS_VLOG(VL_RUNTIME_FRAMEWORK_ACTOR) << "Kernel graph group exit actor:" << GetAID()
                                          << " cost time:" << (end_time - start_time_) * kSecondsToMilliseconds;
    }

    FetchInput(context);
    if (IsRunningFailed(context)) {
      MS_LOG(INFO) << "Run failed and early stop.";
      return;
    }

    // Note that IncreaseNewRefCounts must be in front of SendMemoryFreeReq. SendMemoryFreeReq will decreasing the
    // dynamic ref count. Avoid the illegal timing problem that the dynamic reference count is decremented and then
    // incremented.
    IncreaseNewRefCounts(context);
    SendMemoryFreeReq(context);

    EraseInput(context);
    SendOutput(context);
    MS_VLOG(VL_RUNTIME_FRAMEWORK_ACTOR) << "End run actor:" << GetAID();
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    std::string error_info = "Actor fun failed:" + GetAID().Name();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context), error_info);
  }
  MS_VLOG(VL_RUNTIME_FRAMEWORK_ACTOR) << "Control actor:" << GetAID() << " end run.";
}

void ControlActor::RunOpPartial(const OpPartialPtr &partial, size_t position, OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  (void)input_op_partials_[sequential_num].emplace_back(position, partial);

  auto is_run = CheckRunningCondition(context);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_ACTOR_MSG)
    << "Actor(" << GetAID().Name() << ") receive the input op partial, position:" << position
    << " and check running condition:" << is_run;
  if (is_run) {
    Run(context);
  }
}

void ControlActor::RunBranchID(int branch_id, OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  input_branch_ids_[sequential_num].push(branch_id);

  auto is_run = CheckRunningCondition(context);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_ACTOR_MSG)
    << "Actor(" << GetAID().Name() << ") receive the input branch id and check running condition:" << is_run;
  if (is_run) {
    Run(context);
  }
}

bool ControlActor::CheckRunningCondition(const OpContext<KernelTensor> *context) const {
  MS_EXCEPTION_IF_NULL(context);

  if (!AbstractActor::CheckRunningCondition(context)) {
    return false;
  }

  if (input_partials_num_ != 0) {
    const auto &partial_iter = input_op_partials_.find(context->sequential_num_);
    if (partial_iter == input_op_partials_.end()) {
      return false;
    }
    if (partial_iter->second.size() < input_partials_num_) {
      return false;
    } else if (partial_iter->second.size() > input_partials_num_) {
      MS_LOG(ERROR) << "Invalid input partial num:" << partial_iter->second.size() << " need:" << input_partials_num_
                    << " for actor:" << GetAID();
      return false;
    }
  }
  return true;
}

void ControlActor::FetchParameterInput(OpContext<KernelTensor> *const context) {
  if (!enable_input_optimize_) {
    return;
  }
  // Fetch parameter device tensor from device tensor store.
  for (auto &parameter_index : parameter_indexs_) {
    if (parameter_index.first >= device_contexts_.size()) {
      std::string error_info = "The input index is out of range, need:" + std::to_string(parameter_index.first) +
                               " current:" + std::to_string(device_contexts_.size()) + " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    auto kernel_tensor = FetchParameter(parameter_index.second, GetAID());
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    if (parameter_index.first >= input_kernel_tensors_.size()) {
      std::string error_info = "The input index is out of range, need:" + std::to_string(parameter_index.first) +
                               " current:" + std::to_string(input_kernel_tensors_.size()) +
                               " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    MS_LOG(DEBUG) << "Actor:" << GetAID() << " fetch parameter output index:" << parameter_index.second.second
                  << " inner index:" << parameter_index.second.first.second
                  << " kernel tensor:" << kernel_tensor->ToString();
    input_kernel_tensors_[parameter_index.first] = kernel_tensor;
  }
}

void ControlActor::FetchInput(OpContext<KernelTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, GetAID().Name());
  MS_EXCEPTION_IF_NULL(context);

  // Fetch input device tensor from input data.
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter != input_op_datas_.end()) {
    for (auto &input_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(input_data);
      if (IntToSize(input_data->index_) >= input_kernel_tensors_.size()) {
        std::string error_info = "Invalid index, need:" + std::to_string(input_data->index_) +
                                 " current:" + std::to_string(input_kernel_tensors_.size()) +
                                 " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      MS_EXCEPTION_IF_NULL(input_data->data_);
      input_kernel_tensors_[IntToSize(input_data->index_)] = input_data->data_;
    }
  }

  // Fetch input kernel tensor from local kernel tensor.
  for (auto &local_kernel_tensor : local_kernel_tensors_) {
    MS_EXCEPTION_IF_NULL(local_kernel_tensor.second.first);
    if (local_kernel_tensor.first >= input_kernel_tensors_.size()) {
      std::string error_info = "Invalid local index:" + std::to_string(local_kernel_tensor.first) +
                               " current:" + std::to_string(local_kernel_tensors_.size()) +
                               " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    input_kernel_tensors_[local_kernel_tensor.first] = local_kernel_tensor.second.first;
  }

  // Fetch input device tensor from device tensor store.
  for (auto &device_tensor_store_key : device_tensor_store_keys_) {
    auto kernel_tensors = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second.get());
    if (kernel_tensors.empty()) {
      auto &device_context = device_contexts_[device_tensor_store_key.first];
      MS_EXCEPTION_IF_NULL(device_context);
      MS_EXCEPTION_IF_NULL(device_tensor_store_key.second);
      std::string error_info = GetAID().Name() +
                               " get device tensor store failed: " + device_tensor_store_key.second->DebugString() +
                               ", device type:" + std::to_string(static_cast<int>(device_context->GetDeviceType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }

    if (device_tensor_store_key.first >= input_kernel_tensors_.size()) {
      std::string error_info =
        "The input index is out of range, need:" + std::to_string(device_tensor_store_key.first) +
        " current:" + std::to_string(input_kernel_tensors_.size()) + " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    MS_EXCEPTION_IF_NULL(kernel_tensors[0]);
    input_kernel_tensors_[device_tensor_store_key.first] = kernel_tensors[0];
  }
  FetchParameterInput(context);

  for (size_t i = 0; i < output_data_by_output_index_.size(); ++i) {
    if (output_data_by_output_index_[i].empty()) {
      continue;
    }
    const auto &data = input_kernel_tensors_[i];
    MS_EXCEPTION_IF_NULL(data);
    for (auto &output_data : output_data_by_output_index_[i]) {
      MS_EXCEPTION_IF_NULL(output_data);
      output_data->data_ = data;
    }
  }

  // Fetch input partial from input data.
  const auto &partial_iter = input_op_partials_.find(context->sequential_num_);
  if (partial_iter != input_op_partials_.end()) {
    for (const auto &input_partial : partial_iter->second) {
      if (input_partial.first >= input_partials_.size()) {
        std::string error_info = "Invalid partial index:" + std::to_string(input_partial.first) +
                                 " vector size:" + std::to_string(input_partials_.size()) +
                                 " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      input_partials_[input_partial.first] = input_partial.second;
    }
  }
  // Fetch input partial from local partial.
  for (const auto &local_partial : local_partials_) {
    if (local_partial.first >= input_partials_.size()) {
      std::string error_info = "Invalid partial index:" + std::to_string(local_partial.first) +
                               " vector size:" + std::to_string(input_partials_.size()) +
                               " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    MS_EXCEPTION_IF_NULL(local_partial.second);
    input_partials_[local_partial.first] = local_partial.second;
  }
  // Fetch branch id in stack.
  auto iter = input_branch_ids_.find(context->sequential_num_);
  if (iter != input_branch_ids_.end() && (!iter->second.empty())) {
    output_branch_id_ = iter->second.top();
  }
}

void ControlActor::IncreaseNewRefCounts(OpContext<KernelTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, GetAID().Name());
  MS_EXCEPTION_IF_NULL(context);
  // Increase dynamic ref count by the output data.
  for (size_t i = 0; i < output_data_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(output_data_[i].first);
    if (output_data_[i].first->data_ == nullptr) {
      std::string error_info = GetAID().Name() + " fetches data null, data index:" + std::to_string(i) +
                               " to actor:" + output_data_[i].first->op_id_.Name() +
                               " index:" + std::to_string(output_data_[i].first->index_);
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    const auto &kernel_tensor = output_data_[i].first->data_;
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    if (i < output_need_disable_dynamic_ref_counts_.size() && output_need_disable_dynamic_ref_counts_[i]) {
      MS_LOG(DEBUG) << "Disable dynamic ref count for kernel tensor:" << kernel_tensor
                    << " info:" << kernel_tensor->ToString() << " for actor:" << GetAID();
      kernel_tensor->UpdateFlag(device::kDeviceAddressFlagNullptr);
      continue;
    }
    IncreaseNewRefCount(output_data_[i].first.get());
  }

  // Increase dynamic ref count by the output partial.
  for (const auto &output_partial_arrow : output_partial_arrows_) {
    MS_EXCEPTION_IF_NULL(output_partial_arrow);
    if (IntToSize(output_partial_arrow->from_output_index_) >= input_partials_.size()) {
      std::string error_info = "Invalid partial input:" + std::to_string(output_partial_arrow->from_output_index_) +
                               " current:" + std::to_string(input_partials_.size()) + " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    auto output_partial = input_partials_[IntToSize(output_partial_arrow->from_output_index_)];
    IncreaseNewRefCountForPartial(output_partial);
  }
}

void ControlActor::SendMemoryFreeReq(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  const auto &sequential_num = context->sequential_num_;

  // Collect the input kernel tensors.
  std::vector<KernelTensorPtr> memory_free_list;
  if (input_op_datas_.count(sequential_num) > 0) {
    for (auto &input_op_data : input_op_datas_[sequential_num]) {
      MS_EXCEPTION_IF_NULL(input_op_data);
      MS_EXCEPTION_IF_NULL(input_op_data->data_);
      if (input_need_disable_dynamic_ref_counts_.find(input_op_data->index_) !=
          input_need_disable_dynamic_ref_counts_.end()) {
        MS_LOG(DEBUG) << "Actor:" << GetAID() << " skip free dynamic ref count for:" << input_op_data->data_
                      << " index:" << input_op_data->index_;
        continue;
      }
      (void)memory_free_list.emplace_back(input_op_data->data_);
    }
  }

  if (input_op_partials_.count(sequential_num) > 0) {
    for (auto &input_op_partial : input_op_partials_[sequential_num]) {
      GetAllKernelTensors(input_op_partial.second, &memory_free_list);
    }
  }

  if (memory_free_list.size() > 0) {
    memory_free_lists_.push(memory_free_list);
    if (ActorDispatcher::is_memory_free_sync()) {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &(memory_free_lists_.back()),
                                device_contexts_[0], context, GetAID());
    } else {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &(memory_free_lists_.back()),
                            device_contexts_[0], context, GetAID());
    }
  }
}

void ControlActor::EraseInput(const OpContext<KernelTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  const auto &sequential_num = context->sequential_num_;
  AbstractActor::EraseInput(context);
  if (input_partials_num_ != 0) {
    auto ret = input_op_partials_.erase(sequential_num);
    if (ret == 0) {
      std::string error_info = "Erase input partial failed: " + GetAID().Name();
      // The sequential num may be invalid, can't set the promise value of context.
      MS_LOG(ERROR) << error_info << ", sequential_num: " << sequential_num;
    }
  }

  if (input_branch_ids_.find(sequential_num) != input_branch_ids_.end()) {
    input_branch_ids_[sequential_num].pop();
    if (input_branch_ids_[sequential_num].empty()) {
      auto ret = input_branch_ids_.erase(sequential_num);
      if (ret == 0) {
        MS_LOG(ERROR) << "Erase input branch id failed: " << GetAID() << ", sequential_num: " << sequential_num;
        return;
      }
    }
  }
}

void ControlActor::SendOutput(OpContext<KernelTensor> *const context) {
  for (const auto &pair : created_heter_kernel_tensors_) {
    created_kernel_tensors_.emplace_back(pair.second);
  }
  created_heter_kernel_tensors_.clear();
  // Send branch id.
  for (const auto &branch_id_arrow : output_branch_id_arrows_) {
    ActorDispatcher::Send(branch_id_arrow, &ControlActor::RunBranchID, output_branch_id_, context);
  }

  // Send data in base class.
  AbstractActor::SendOutput(context);

  // Send Partial.
  for (const auto &partial_arrow : output_partial_arrows_) {
    MS_EXCEPTION_IF_NULL(partial_arrow);
    if (IntToSize(partial_arrow->from_output_index_) >= input_partials_.size()) {
      std::string error_info = "Invalid partial input:" + std::to_string(partial_arrow->from_output_index_) +
                               " current:" + std::to_string(input_partials_.size()) + " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    auto output_partial = input_partials_[IntToSize(partial_arrow->from_output_index_)];
    MS_EXCEPTION_IF_NULL(output_partial);
    ActorDispatcher::Send(partial_arrow->to_op_id_, &ControlActor::RunOpPartial, output_partial,
                          IntToSize(partial_arrow->to_input_index_), context);
  }

  // Update the start time in end actor.
  for (const auto &actor : end_actors_) {
    MS_EXCEPTION_IF_NULL(actor);
    actor->set_start_time(GetTime());
  }
}
namespace {
CNodePtr CreateRealMakeTuple(const std::vector<KernelTensor *> &addr_list, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimRealMakeTuple)};
  auto new_cnode = func_graph->NewCNode(inputs);
  std::vector<std::string> formats;
  MS_EXCEPTION_IF_NULL(new_cnode);
  std::vector<abstract::AbstractBasePtr> abs_list;
  for (const auto &addr_kernel : addr_list) {
    MS_EXCEPTION_IF_NULL(addr_kernel);
    auto abs =
      std::make_shared<abstract::AbstractTensor>(TypeIdToType(addr_kernel->dtype_id()), addr_kernel->GetShapeVector());
    abs_list.emplace_back(abs);
    formats.emplace_back(kernel::GetFormatFromEnumToStr(addr_kernel->format()));
    MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Create new abstract:" << abs->ToString();
  }
  auto tuple_abs = std::make_shared<abstract::AbstractTuple>(abs_list);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Create abstract for real make tuple:" << tuple_abs->ToString();
  // Set dynamic len element abstract to check the abstract is dynamic len.
  abstract::AbstractBasePtr element_abs = (abs_list.empty() ? std::make_shared<abstract::AbstractTensor>(
                                                                TypeIdToType(TypeId::kNumberTypeInt64), ShapeVector())
                                                            : abs_list[0]);
  tuple_abs->set_dynamic_len_element_abs(element_abs);
  new_cnode->set_abstract(tuple_abs);

  // Create kernel info for node and set format for it.
  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(builder);
  kernel_info->set_select_kernel_build_info(builder->Build());
  new_cnode->set_kernel_info(kernel_info);
  builder->SetOutputsFormat(formats);
  return new_cnode;
}

void CheckDeviceAddressConsist(OpContext<KernelTensor> *const context, const std::vector<KernelTensor *> &addr_list,
                               const std::string &actor_name) {
  MS_EXCEPTION_IF_NULL(context);
  if (addr_list.empty() || addr_list[0] == nullptr) {
    return;
  }
  // Check consistence of device address.
  const auto &shape = addr_list[0]->GetShapeVector();
  const auto &size = addr_list[0]->device_address()->GetSize();
  const auto &type = addr_list[0]->dtype_id();
  for (size_t i = 1; i < addr_list.size(); ++i) {
    MS_EXCEPTION_IF_NULL(addr_list[i]);
    MS_EXCEPTION_IF_NULL(addr_list[i]->device_address());
    if (size != addr_list[i]->device_address()->GetSize() || type != addr_list[i]->dtype_id()) {
      MS_LOG(ERROR) << "Failed to merge two device address, addr1:" << addr_list[0]->ToString()
                    << " addr2:" << addr_list[i]->ToString() << " for actor:" << actor_name;
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "Failed to merge two device address");
    }
    if (shape != addr_list[i]->GetShapeVector()) {
      MS_LOG(WARNING) << "Merge two device address with different shape, addr1 shape:" << shape
                      << " addr2 shape:" << addr_list[i]->GetShapeVector() << " for actor:" << actor_name;
    }
  }
}
}  // namespace

void ControlActor::MergeDeviceAddress(OpContext<KernelTensor> *const context,
                                      const std::vector<KernelTensor *> &addr_list, KernelTensorPtr *kernel_tensor) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  if (addr_list.empty()) {
    MergeEmptyAddressDeviceAddress(context, addr_list, kernel_tensor);
    return;
  }

  CheckDeviceAddressConsist(context, addr_list, GetAID().Name());
  MS_EXCEPTION_IF_NULL(addr_list[0]);
  MS_EXCEPTION_IF_NULL(addr_list[0]->device_address());
  const auto &total_size = addr_list[0]->device_address()->GetSize() * addr_list.size();
  ShapeVector total_shape = {SizeToLong(addr_list.size())};
  const auto &shape = addr_list[0]->GetShapeVector();
  total_shape.insert(total_shape.end(), shape.begin(), shape.end());
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {addr_list[0]->device_address()->GetDeviceType(), addr_list[0]->device_address()->device_id()});
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  const auto device_name = device::GetDeviceNameByType(device_context->device_context_key().device_type_);

  abstract::BaseShapePtrList shape_list(addr_list.size(), addr_list[0]->GetShape());
  auto tuple_shape = std::make_shared<abstract::TupleShape>(shape_list);
  TypePtrList type_list(addr_list.size(), addr_list[0]->GetType());
  auto tuple_type = std::make_shared<Tuple>(type_list);
  MS_LOG(DEBUG) << "Create kernel tensor by shape:" << tuple_shape->ToString() << " type:" << tuple_type->ToString()
                << " in device address:" << addr_list[0]->device_address();
  const auto &new_kernel_tensor = AnfAlgo::CreateKernelTensor(
    tuple_shape, tuple_type, nullptr, nullptr, total_size, kernel::GetFormatFromEnumToStr(addr_list[0]->format()),
    addr_list[0]->dtype_id(), total_shape, device_name, device_context->device_context_key().device_id_);
  new_kernel_tensor->set_stream_id(addr_list[0]->device_address()->stream_id());
  const auto &new_device_tensor = new_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(new_device_tensor);

  MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "Create kernel tensor:" << new_kernel_tensor->ToString();
  if (!device_context->device_res_manager_->AllocateMemory(new_device_tensor.get(), kDefaultStreamIndex)) {
    SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(GraphExecutionStrategy::kPipeline, *context, *device_context,
                                                GetAID().Name(), new_device_tensor->GetSize());
  }
  static std::string name = "Alloc memory";
  new_kernel_tensor->IncreaseNewRefCount(name);
  MS_EXCEPTION_IF_NULL(new_device_tensor->GetMutablePtr());

  // Create a new real maketuple node for new device address.
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  auto new_cnode = CreateRealMakeTuple(addr_list, fg);
  AnfAlgo::SetOutputKernelTensor(new_kernel_tensor, 0, new_cnode.get());
  created_new_graphs_.emplace_back(fg);
  created_new_nodes_.emplace_back(new_cnode);
  new_device_tensor->SetNodeIndex(new_cnode, 0);
  new_device_tensor->set_from_persistent_mem(addr_list[0]->device_address()->from_persistent_mem());

  // Merge device address list into a single device address.
  auto tmp_kernel_tensor = AnfAlgo::CreateKernelTensor(
    new_device_tensor->GetMutablePtr(), addr_list[0]->device_address()->GetSize(), addr_list[0]->format(),
    addr_list[0]->dtype_id(), shape, device_name, device_context->device_context_key().device_id_);
  tmp_kernel_tensor->set_stream_id(addr_list[0]->device_address()->stream_id());
  const auto &tmp_device_tensor = tmp_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(tmp_device_tensor);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "Create kernel tensor:" << tmp_kernel_tensor->ToString();
  std::shared_ptr<int64_t> max_task_id_on_stream = nullptr;
  for (size_t i = 0; i < addr_list.size(); ++i) {
    auto task_id_on_stream = addr_list[i]->task_id_on_stream();
    if (task_id_on_stream != nullptr) {
      if (max_task_id_on_stream == nullptr) {
        max_task_id_on_stream = task_id_on_stream;
      } else {
        if (*max_task_id_on_stream < *task_id_on_stream) {
          max_task_id_on_stream = task_id_on_stream;
        }
      }
    }
    if (!SyncAllStreamForDeviceAddress(tmp_device_tensor, addr_list[i]->device_address()) ||
        !SyncCopy(tmp_kernel_tensor.get(), addr_list[i], kDefaultStreamIndex)) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR(*context, "Sync device to device failed.");
    }
    tmp_device_tensor->set_ptr((reinterpret_cast<char *>(tmp_device_tensor->GetMutablePtr())) +
                               addr_list[0]->device_address()->GetSize());
  }
  new_kernel_tensor->set_task_id_on_stream(max_task_id_on_stream);
  tmp_device_tensor->set_ptr(nullptr);
  created_kernel_tensors_.emplace_back(new_kernel_tensor);
  MS_LOG(DEBUG) << "actor:" << GetAID() << " create new kernel tensor:" << new_kernel_tensor->ToString()
                << " for addr list size:" << addr_list.size();
  (*kernel_tensor) = new_kernel_tensor;
  return;
}

void ControlActor::MergeEmptyAddressDeviceAddress(OpContext<KernelTensor> *const context,
                                                  const std::vector<KernelTensor *> &addr_list,
                                                  KernelTensorPtr *kernel_tensor) {
  // Create device address for empty tuple.
  // Fetch the default device context for empty sequence.
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device::GetDeviceTypeByName(context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET)),
     context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);

  auto tuple_shape = std::make_shared<abstract::TupleShape>();
  auto tuple_type = std::make_shared<Tuple>();
  const auto &new_kernel_tensor = AnfAlgo::CreateKernelTensor(
    tuple_shape, tuple_type, nullptr, nullptr, 0, kOpFormat_DEFAULT, TypeId::kNumberTypeInt64, ShapeVector(),
    device::GetDeviceNameByType(device_context->device_context_key().device_type_),
    device_context->device_context_key().device_id_);
  const auto &new_device_tensor = new_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(new_device_tensor);
  if (!device_context->device_res_manager_->AllocateMemory(new_device_tensor.get(), kDefaultStreamIndex)) {
    SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(GraphExecutionStrategy::kPipeline, *context, *device_context,
                                                GetAID().Name(), new_device_tensor->GetSize());
  }
  static std::string name = "Alloc memory";
  new_kernel_tensor->IncreaseNewRefCount(name);
  created_kernel_tensors_.emplace_back(new_kernel_tensor);
  (*kernel_tensor) = new_kernel_tensor;
  MS_LOG(DEBUG) << "actor:" << GetAID() << " create new kernel tensor:" << new_kernel_tensor->ToString()
                << " for empty addr list";
}

void ControlActor::ResetState(OpContext<KernelTensor> *const context) {
  MS_LOG(INFO) << "Start free control actor " << GetAID();
  while (!memory_free_lists_.empty()) {
    auto kernel_tensors = memory_free_lists_.front();
    memory_free_lists_.pop();
    MS_LOG(INFO) << "device tensors size: " << kernel_tensors.size();
    for (auto kernel_tensor : kernel_tensors) {
      if (kernel_tensor == nullptr || kernel_tensor->device_address() == nullptr ||
          kernel_tensor->device_address()->GetPtr() == nullptr) {
        continue;
      }
      auto device_tensor = kernel_tensor->device_address().get();
      // Weight can not be free.
      if (kernel_tensor->new_ref_count() == SIZE_MAX) {
        continue;
      }
      auto held_by_nodes = device_tensor->held_by_nodes();
      if (!held_by_nodes.empty()) {
        FreeMemoryByValueNode(held_by_nodes, device_tensor);
        kernel_tensor->set_new_ref_count(0);
        continue;
      }
      const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
        {device_tensor->GetDeviceType(), device_tensor->device_id()});
      MS_EXCEPTION_IF_NULL(device_context);
      FreeMemoryByDeviceContext(device_tensor, device_context);
      kernel_tensor->set_new_ref_count(0);
    }
  }
  MS_LOG(INFO) << "End free control actor " << GetAID();
}
}  // namespace runtime
}  // namespace mindspore
