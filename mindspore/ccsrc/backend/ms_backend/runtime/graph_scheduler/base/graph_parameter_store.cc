/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "backend/ms_backend/runtime/graph_scheduler/base/graph_parameter_store.h"
#include <vector>
#include <utility>
#include <algorithm>
#include <string>
#include "backend/ms_backend/runtime/graph_scheduler/heterogeneous/device_tensor_copy_store.h"
#include "backend/ms_backend/runtime/actors/base/actor_common.h"
#include "backend/ms_backend/runtime/graph_executor/kernel_capture/graph_capture_manager.h"
#include "include/backend/common/device_address_utils.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "utils/ms_context.h"
#include "utils/llm_manager.h"

namespace mindspore {
namespace runtime {
void GraphParameterStore::SetPositionWeight(size_t outer_index, bool is_weight) {
  if (outer_index >= is_weights_.size()) {
    MS_LOG(EXCEPTION) << "Outer index is larger than the size of is weights [" << is_weights_.size() << "].";
  }

  if (!is_weights_[outer_index] && is_weight) {
    weight_num_++;
  }
  is_weights_[outer_index] = is_weight;
}

bool GraphParameterStore::GetPositionWeight(size_t outer_index) {
  if (outer_index >= is_weights_.size()) {
    MS_LOG(ERROR) << "Index " << outer_index << ", is out of range of outer size: " << is_weights_.size();
    return false;
  }
  return is_weights_[outer_index];
}

void GraphParameterStore::SetDeviceTensorPrepared(size_t outer_idx, size_t inner_idx, bool is_prepared) {
  auto &kernel_tensor_with_info = parameter_kernel_tensors_[outer_idx][inner_idx];
  kernel_tensor_with_info.second.second = is_prepared;
}

bool GraphParameterStore::GetDeviceTensorPrepared(size_t outer_idx, size_t inner_idx) {
  auto &kernel_tensor_with_info = parameter_kernel_tensors_[outer_idx][inner_idx];
  return kernel_tensor_with_info.second.second;
}

size_t GraphParameterStore::GetNonWeightParameterNum() { return non_weight_data_num_; }

void GraphParameterStore::SetPositionTensor(size_t outer_index, bool is_tensor) {
  if (outer_index >= is_tensors_.size()) {
    MS_LOG(ERROR) << "Index " << outer_index << ", is out of range of outer size: " << is_tensors_.size();
    return;
  }
  is_tensors_[outer_index] = is_tensor;
}

bool GraphParameterStore::GetPositionTensor(size_t outer_index) { return is_tensors_[outer_index]; }

void GraphParameterStore::SetParameterUsedTimes(size_t outer_index, size_t inner_index, size_t times) {
  parameter_used_times_[outer_index][inner_index] = times;
}

bool GraphParameterStore::GetOffloaded(size_t outer_index, size_t inner_index) {
  return is_offload_parameter_[outer_index][inner_index];
}

void GraphParameterStore::SetOffloaded(size_t outer_index, size_t inner_index, bool is_offload) {
  is_offload_parameter_[outer_index][inner_index] = is_offload;
}

bool GraphParameterStore::GetPinned(size_t outer_index, size_t inner_index) {
  return is_parameter_pinned_[outer_index][inner_index];
}

void GraphParameterStore::SetPinned(size_t outer_index, size_t inner_index, bool is_offload) {
  is_parameter_pinned_[outer_index][inner_index] = is_offload;
}

bool GraphParameterStore::IsConcurrentlyUse(size_t outer_index, size_t inner_index) {
  return parameter_used_times_[outer_index][inner_index] > 1;
}

void GraphParameterStore::SetFrontNodeToIndex(AnfNode *node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &iter = front_node_to_index_.find(node);
  if (iter != front_node_to_index_.end()) {
    MS_LOG(INFO) << "Update index for front node " << node->DebugString() << " in graph parameter store.";
    iter->second = index;
  }
  front_node_to_index_.emplace(node, index);
  index_to_front_node_.emplace(index, node);
}

void GraphParameterStore::InsertDeviceTensorIntoCallback(const DeviceAddressPtr &device_tensor) {
  std::unique_lock<std::shared_mutex> lock(param_mutex_);
  device_tensor_in_callback_.push_back(device_tensor);
}

void GraphParameterStore::InsertNonWeightRefMaxInputs(size_t outer_index, size_t inner_index) {
  std::unique_lock<std::shared_mutex> lock(param_mutex_);
  non_weight_ref_max_inputs_.emplace(outer_index, inner_index);
}

void GraphParameterStore::ResetPrepareState() {
  for (size_t i = 0; i < parameter_kernel_tensors_.size(); ++i) {
    auto &kernel_tensors = parameter_kernel_tensors_[i];
    for (size_t j = 0; j < kernel_tensors.size(); ++j) {
      kernel_tensors[j].second.second = false;
    }
  }
  device_tensor_in_callback_.reserve(buffer_size_);
}

void GraphParameterStore::ResetAddrRefCount(size_t outer_index, size_t inner_index) {
  std::unique_lock<std::shared_mutex> lock(param_mutex_);
  auto &kernel_tensor_with_info = parameter_kernel_tensors_[outer_index][inner_index];
  bool is_ref_count_max = kernel_tensor_with_info.second.first == SIZE_MAX;

  if (kernel_tensor_with_info.first != nullptr) {
    auto &kernel_tensor = kernel_tensor_with_info.first;
    auto user_cnt = kernel_tensor_with_info.second.first;
    if (user_cnt > 0) {
      // When allocate memory, the ref count would be increase, so it should be decrease here.
      if (is_ref_count_max) {
        kernel_tensor->set_new_ref_count(SIZE_MAX);
      } else {
        static std::string name = "Parameter store";
        kernel_tensor->IncreaseNewRefCount(name, user_cnt - 1);
      }
      kernel_tensor_with_info.first->ClearFlag(device::kDeviceAddressFlagNotUsed);
      MS_LOG(DEBUG) << "Parameter store set new ref count:" << (user_cnt - 1)
                    << " for kernel tensor:" << kernel_tensor_with_info.first->ToString();
    } else {
      MS_LOG(DEBUG) << "User count:0 for parameter store outer index:" << outer_index << " inner index:" << inner_index
                    << " for kernel tensor:" << kernel_tensor->ToString();
    }
    return;
  }
}

KernelTensorPtr GraphParameterStore::Fetch(size_t outer_index, size_t inner_index) {
  std::shared_lock<std::shared_mutex> lock(param_mutex_);
  const auto &kernel_tensor_with_info = parameter_kernel_tensors_[outer_index][inner_index];
  return kernel_tensor_with_info.first;
}

const std::function<void(size_t)> &GraphParameterStore::GetAsyncMemcpyFun(size_t outer_index,
                                                                          size_t inner_index) const {
  return async_copy_funcs_[outer_index][inner_index];
}

void GraphParameterStore::SetAsyncMemcpyFun(size_t outer_index, size_t inner_index,
                                            std::function<void(size_t)> &&func) {
  async_copy_funcs_[outer_index][inner_index] = std::move(func);
}

void GraphParameterStore::Push(size_t outer_index, size_t inner_index, const KernelTensorPtr &value, size_t cnt) {
  CheckIndexValid(outer_index, inner_index);
  std::unique_lock<std::shared_mutex> lock(param_mutex_);
  MS_LOG(DEBUG) << "Push parameter store for outer index:" << outer_index << " inner index:" << inner_index
                << " count:" << cnt << " kernel tensor:" << value;
  auto &kernel_tensor_with_info = parameter_kernel_tensors_[outer_index][inner_index];
  // Record non weight parameter num in store.
  if (!is_weights_[outer_index] && kernel_tensor_with_info.first == nullptr) {
    non_weight_data_num_++;
  }
  kernel_tensor_with_info.first = value;
  kernel_tensor_with_info.second.first = cnt;
  if (value->device_address()) {
    parameter_device_types_[outer_index][inner_index] = value->device_address()->GetDeviceType();
    MS_LOG(DEBUG) << "Set graph parameter name:"
                  << device::GetDeviceNameByType(value->device_address()->GetDeviceType())
                  << " outer index:" << outer_index << " inner index:" << inner_index;
  }
}

device::DeviceType GraphParameterStore::GetParameterDeviceType(size_t outer_index, size_t inner_index) const {
  return parameter_device_types_[outer_index][inner_index];
}

bool GraphParameterStore::CheckBufferSize(size_t outer_index) const {
  std::shared_lock<std::shared_mutex> lock(param_mutex_);
  return buffers_[outer_index].size() > 0;
}

Tensor *GraphParameterStore::FetchTensor(size_t args_index, const KernelWithIndex &node) {
  // Process tensor types that are frequently used to speed up fetchtensor performance.
  if (is_tensors_[args_index]) {
    auto tensor = utils::cast<tensor::TensorPtr>((*input_args_)[args_index]);
    MS_EXCEPTION_IF_NULL(tensor);
    return tensor.get();
  }

  if (args_index >= buffers_.size()) {
    MS_LOG(EXCEPTION) << "Index " << args_index << " is out of buffers range " << buffers_.size() << ".";
  }
  Tensor *tensor = nullptr;
  if (CheckBufferSize(args_index)) {
    std::shared_lock<std::shared_mutex> lock(param_mutex_);
    if (node.second >= buffers_[args_index].size()) {
      MS_LOG(EXCEPTION) << "Node position " << node.second << " is out of buffers position "
                        << buffers_[args_index].size() << " range.";
    }
    tensor = buffers_[args_index][node.second].get();
  } else {
    tensor = FlattenInputTensorByArg(args_index, node);
  }
  MS_EXCEPTION_IF_NULL(tensor);
  return tensor;
}

bool GraphParameterStore::RecordGraphInputsAndIsDyn(const GraphCompilerInfo *graph_compiler_info,
                                                    const std::vector<size_t> &input_index,
                                                    const std::vector<AnfNodePtr> &parameters) {
  bool isDyn = false;
  auto &llm_manager = LLMManager::GetInstance();

  for (size_t l = 0; l < input_index.size(); ++l) {
    auto i = input_index[l];
    auto origin_parameter = parameters[l];
    MS_EXCEPTION_IF_NULL(origin_parameter);
    if (i >= buffers_.size()) {
      MS_LOG(EXCEPTION) << "Index " << i << " is out of range of buffers size: " << buffers_.size();
    }

    const auto &input_tensor = buffers_[i][0];
    if (input_tensor == nullptr) {
      MS_LOG(ERROR) << "The input tensor is nullptr for arg outer index: " << i;
      continue;
    }
    if (!isDyn) {
      if (host_tensors_shape_[i] != input_tensor->shape() || input_tensor->shape().empty()) {
        isDyn = true;
      }
    }
    host_tensors_shape_[i] = input_tensor->shape();
    input_tensor->set_name(origin_parameter->fullname_with_scope());
    MS_LOG(DEBUG) << "Add graph input: " << origin_parameter->fullname_with_scope();
    llm_manager.add_graph_input(origin_parameter->fullname_with_scope(), input_tensor);
    MS_LOG(DEBUG) << "Add input tensor data for input parameter: " << origin_parameter->fullname_with_scope();
  }
  return isDyn;
}

void GraphParameterStore::ConvertNormalInputContiguous(const std::vector<size_t> &input_index) {
  for (size_t l = 0; l < input_index.size(); ++l) {
    auto i = input_index[l];
    if (i >= buffers_.size()) {
      MS_LOG(EXCEPTION) << "Index " << i << " is out of range of buffers size: " << buffers_.size();
    }
    auto buffer_inner_size = buffers_[i].size();
    if (buffer_inner_size == 1) {
      const auto &input_tensor = buffers_[i][0];
      DeviceAddressUtils::ConvertContiguousTensorSync(input_tensor, 0);
    }
  }
}

KernelTensorPtr GraphParameterStore::GetReleasedCheckInfo(size_t outer_index, size_t inner_index) {
  CheckIndexValid(outer_index, inner_index);
  return released_check_addresses_[outer_index][inner_index];
}

void AddCopyDataCallBack(const std::vector<DeviceAddressPtr> &device_tensor_in_callback) {
  if (device_tensor_in_callback.empty()) {
    return;
  }
  device::CallbackFunc callback_func = [device_tensor_in_callback]() {
    // Clear buffer automatically.
  };

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  host_context->device_res_manager_->BindDeviceToCurrentThread(false);
  auto callback_ret = host_context->device_res_manager_->LaunchCallback(callback_func, kDefaultStreamIndex);
  if (!callback_ret) {
    MS_LOG(EXCEPTION) << "Async Copy memory launch callback failed";
  }
}

void GraphParameterStore::ReleaseData() {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kReleaseResource, "GraphParameterStore");
  // Add copy data callback to avoid release data before async copy finished.
  std::unique_lock<std::shared_mutex> lock(param_mutex_);
  AddCopyDataCallBack(device_tensor_in_callback_);
  device_tensor_in_callback_.clear();

  static bool sync_copy_input = runtime::IsEnableRuntimeConfig(runtime::kRuntimeSyncCopyInput) ||
                                runtime::RuntimeConf::GetInstance()->launch_blocking();

  for (auto index : non_weight_ref_max_inputs_) {
    std::pair<size_t, size_t> position{index.first, index.second};
    auto &kernel_tensor_with_info = parameter_kernel_tensors_[index.first][index.second];
    auto &kernel_tensor = kernel_tensor_with_info.first;
    if (kernel_tensor != nullptr) {
      auto &device_tensor = kernel_tensor->device_address();
      if (device_tensor != nullptr && kernel_tensor->new_ref_count() == SIZE_MAX &&
          !kernel_tensor->is_ptr_persisted()) {
        MS_LOG(DEBUG) << "Set store device tensor: " << device_tensor.get() << " ptr null, outer idx: " << index.first
                      << ", inner idx: " << index.second << ", info: " << kernel_tensor->ToString();
        // Record released info, so that it can check input next step.
        // Used for CheckInputSize for fetching parameter.
        if (sync_copy_input) {
          auto new_kernel_tensor = kernel_tensor->CloneKernelTensor();
          new_kernel_tensor->set_device_ptr(nullptr);
          released_check_addresses_[index.first][index.second] = new_kernel_tensor;
        }
        release_data_info_[{position}] = {kernel_tensor->GetType(), device_tensor->GetNodeIndex()};
        kernel_tensor->set_device_address(nullptr);
        kernel_tensor->set_tensor_storage_info(nullptr);
      }
    }
  }

  for (auto &buffer : buffers_) {
    buffer.clear();
  }
  buffers_.clear();

  input_args_ = nullptr;
}

void GraphParameterStore::FillBuffer(size_t idx, const std::vector<TensorPtr> &tensors) {
  if (idx >= (*input_args_).size()) {
    MS_LOG(EXCEPTION) << "Index is out of buffer range.";
  }
  if (buffers_[idx].size() > 0) {
    return;
  }
  buffers_[idx] = tensors;
}

std::pair<bool, std::pair<TypePtr, KernelWithIndex>> GraphParameterStore::GetReleasePositionInfo(
  const DeviceTensorPosition &position) {
  std::shared_lock<std::shared_mutex> lock(param_mutex_);
  const auto &iter = release_data_info_.find({position});
  if (iter == release_data_info_.end()) {
    MS_LOG(INFO) << "Can not find type in store, where outer index: " << position.first
                 << ", inner index: " << position.second;
    return std::make_pair(false, std::make_pair(nullptr, std::make_pair(nullptr, 0)));
  }
  return std::make_pair(true, iter->second);
}

Tensor *GraphParameterStore::FlattenInputTensorByArg(size_t arg_index, const KernelWithIndex &front_node) {
  if (arg_index >= (*input_args_).size()) {
    MS_LOG(INFO) << "Arg index out of args range, index is " << arg_index << " and args size is "
                 << (*input_args_).size();
    return nullptr;
  }
  std::unique_lock<std::shared_mutex> lock(param_mutex_);
  tensor::TensorPtr tensor = nullptr;
  if (GetPositionTensor(arg_index)) {
    tensor = utils::cast<tensor::TensorPtr>((*input_args_)[arg_index]);
    MS_EXCEPTION_IF_NULL(tensor);
    buffers_[arg_index].emplace_back(tensor);
  } else {
    AnfAlgo::FlattenInputArg((*input_args_)[arg_index], front_node.first, &(buffers_[arg_index]));
    auto input_tensor_index = FetchInputTensorIndex(front_node);
    if (input_tensor_index >= buffers_[arg_index].size()) {
      MS_LOG(INFO) << "Input tensor index out of args range, index is " << input_tensor_index << " and tensors size is "
                   << buffers_[arg_index].size();
      return nullptr;
    }
    tensor = buffers_[arg_index][input_tensor_index];
  }

  return tensor.get();
}

void GraphParameterStore::Clear() {
  std::unique_lock<std::shared_mutex> lock(param_mutex_);
  parameter_kernel_tensors_.clear();
  release_data_info_.clear();
  front_node_to_index_.clear();
  node_to_real_front_node_.clear();
  index_to_front_node_.clear();
  device_tensor_in_callback_.clear();
  for (auto &buffer : buffers_) {
    buffer.clear();
  }
  buffers_.clear();
}
}  // namespace runtime
}  // namespace mindspore
