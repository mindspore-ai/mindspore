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

#ifndef MINDSPORE_CCSRC_RUNTIME_CORE_GRAPH_SCHEDULER_GRAPH_PARAMETER_STORE_H_
#define MINDSPORE_CCSRC_RUNTIME_CORE_GRAPH_SCHEDULER_GRAPH_PARAMETER_STORE_H_

#include <memory>
#include <map>
#include <set>
#include <vector>
#include <utility>
#include <string>
#include <shared_mutex>
#include "utils/ms_utils.h"
#include "include/backend/visible.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/graph_compiler.h"
namespace mindspore {
namespace runtime {
using mindspore::tensor::Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
using TensorDataPtr = std::shared_ptr<mindspore::tensor::TensorData>;
using DeviceTensor = mindspore::device::DeviceAddress;
using DeviceTensorType = mindspore::device::DeviceType;
using DeviceTensorPtr = std::shared_ptr<DeviceTensor>;
using KernelWithIndex = std::pair<AnfNodePtr, size_t>;
using KernelTensorPtr = kernel::KernelTensorPtr;
using UserCntWithPrepared = std::pair<size_t, bool>;
using DeviceTensorPosition = std::pair<size_t, size_t>;
// The kernel tensor mainly includes address ptr, size and reference count,
// which represents the basic data structure of kernel launch and transfers between actors.
// The args are input from the front every step.
// The parameter kernel tensors (such as weight and non-weight parameter) and the args are save in
// the store, so they can be known by actors and be used for preparing data in actor.
class BACKEND_EXPORT GraphParameterStore {
 public:
  GraphParameterStore() = default;
  ~GraphParameterStore() = default;

  void Resize(size_t front_parameter_size) {
    parameter_kernel_tensors_.resize(front_parameter_size);
    async_copy_funcs_.resize(front_parameter_size);
    is_dynamic_.resize(front_parameter_size);
    is_weights_.resize(front_parameter_size, false);
    is_tensors_.resize(front_parameter_size, false);
    parameter_used_times_.resize(front_parameter_size);
    parameter_device_types_.resize(front_parameter_size);
    is_offload_parameter_.resize(front_parameter_size);
    is_parameter_pinned_.resize(front_parameter_size);
    released_check_addresses_.resize(front_parameter_size);
  }

  void ResizePosition(size_t outer_index, size_t tuple_unfold_length) {
    if (outer_index >= parameter_kernel_tensors_.size()) {
      MS_LOG(EXCEPTION) << "inner index is larger than the size of parameter kernel tensors [" << outer_index << "].";
    }
    parameter_kernel_tensors_[outer_index].resize(tuple_unfold_length);
    async_copy_funcs_[outer_index].resize(tuple_unfold_length, nullptr);
    is_dynamic_[outer_index].resize(tuple_unfold_length, false);
    parameter_used_times_[outer_index].resize(tuple_unfold_length, 0);
    parameter_device_types_[outer_index].resize(tuple_unfold_length);
    is_offload_parameter_[outer_index].resize(tuple_unfold_length, false);
    is_parameter_pinned_[outer_index].resize(tuple_unfold_length, false);
    released_check_addresses_[outer_index].resize(tuple_unfold_length);
    buffer_size_ += tuple_unfold_length;
  }

  void CheckIndexValid(size_t outer_index, size_t inner_index) const {
    if (outer_index >= parameter_kernel_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Outer index is larger than the size of parameter kernel tensors ["
                        << parameter_kernel_tensors_.size() << "].";
    }
    if (inner_index >= parameter_kernel_tensors_[outer_index].size()) {
      MS_LOG(EXCEPTION) << "inner index is larger than the size of parameter kernel tensors ["
                        << parameter_kernel_tensors_[outer_index].size() << "].";
    }
  }

  void SetInputArgs(const VectorRef &args) {
    input_args_ = const_cast<VectorRef *>(&args);
    buffers_.resize(args.size());
    host_tensors_shape_.resize(args.size());
  }
  VectorRef *GetInputArgs() const { return input_args_; }

  void SetDeviceTensorPrepared(size_t outer_idx, size_t inner_idx, bool is_prepared);
  bool GetDeviceTensorPrepared(size_t outer_idx, size_t inner_idx);

  void SetUserCnt(size_t outer_idx, size_t inner_idx, size_t cnt) {
    auto &kernel_tensor_with_info = parameter_kernel_tensors_[outer_idx][inner_idx];
    MS_LOG(DEBUG) << "Set use count:" << cnt << " for parameter store outer index:" << outer_idx
                  << " inner index:" << inner_idx;
    kernel_tensor_with_info.second.first = cnt;
    parameter_used_times_[outer_idx][inner_idx]++;
  }

  void IncreaseUserCnt(size_t outer_idx, size_t inner_idx) {
    auto &kernel_tensor_with_info = parameter_kernel_tensors_[outer_idx][inner_idx];
    if (kernel_tensor_with_info.second.first != SIZE_MAX) {
      kernel_tensor_with_info.second.first++;
      MS_LOG(DEBUG) << "Increase use count:" << kernel_tensor_with_info.second.first
                    << " for parameter store outer index:" << outer_idx << " inner index:" << inner_idx;
    }
    parameter_used_times_[outer_idx][inner_idx]++;
  }

  size_t GetUserCnt(size_t outer_idx, size_t inner_idx) {
    auto &kernel_tensor_with_info = parameter_kernel_tensors_[outer_idx][inner_idx];
    return kernel_tensor_with_info.second.first;
  }

  void SetFrontNodeToIndex(AnfNode *node, size_t index);

  size_t GetFrontNodeToIndex(AnfNode *node) {
    MS_EXCEPTION_IF_NULL(node);
    auto iter = front_node_to_index_.find(node);
    if (iter == front_node_to_index_.end()) {
      MS_LOG(EXCEPTION) << "Can not find index for front node " << node->DebugString() << " in graph parameter store.";
    }
    return iter->second;
  }

  void CorrectFrontNodeMap(const KernelWithIndex &node_with_index, const KernelWithIndex &real_node_with_index) {
    MS_EXCEPTION_IF_NULL(node_with_index.first);
    MS_EXCEPTION_IF_NULL(real_node_with_index.first);
    const auto &iter = node_to_real_front_node_.find(node_with_index);
    if (iter != node_to_real_front_node_.end()) {
      MS_LOG(INFO) << "Node: " << node_with_index.first->DebugString() << ", index: " << node_with_index.second
                   << ", is already map to real front node: " << real_node_with_index.first->DebugString()
                   << ", index: " << real_node_with_index.second << " in graph parameter store.";
      return;
    }
    node_to_real_front_node_.emplace(node_with_index, real_node_with_index);
  }
  KernelWithIndex GetRealFrontNode(const KernelWithIndex &node_with_index) {
    MS_EXCEPTION_IF_NULL(node_with_index.first);
    auto iter = node_to_real_front_node_.find(node_with_index);
    if (iter == node_to_real_front_node_.end()) {
      MS_LOG(EXCEPTION) << "Can not find real front node for node " << node_with_index.first->DebugString()
                        << ", index: " << node_with_index.second << " in graph parameter store.";
    }
    return iter->second;
  }

  bool IsFrontNodeInStore(AnfNode *node) {
    auto iter = front_node_to_index_.find(node);
    if (iter == front_node_to_index_.end()) {
      return false;
    }
    return true;
  }

  void SetIsPositionDynamic(size_t outer_index, size_t inner_index, bool is_dynamic) {
    CheckIndexValid(outer_index, inner_index);
    is_dynamic_[outer_index][inner_index] = is_dynamic;
  }

  bool IsPositionDynamic(size_t outer_index, size_t inner_index) { return is_dynamic_[outer_index][inner_index]; }

  void InsertNonWeightRefMaxInputs(size_t outer_index, size_t inner_index);

  // Reset the prepare state at the step beginning.
  void ResetPrepareState();

  void ResetAddrRefCount(size_t outer_index, size_t inner_index);

  // Fetch kernel tensor with index from parameter_kernel_tensors_.
  KernelTensorPtr Fetch(size_t outer_index, size_t inner_index);

  // Push the kernel tensor and user count to parameter_kernel_tensors_.
  void Push(size_t outer_index, size_t inner_index, const KernelTensorPtr &value, size_t cnt);
  const std::function<void(size_t)> &GetAsyncMemcpyFun(size_t outer_index, size_t inner_index) const;
  void SetAsyncMemcpyFun(size_t outer_index, size_t inner_index, std::function<void(size_t)> &&func);

  device::DeviceType GetParameterDeviceType(size_t outer_index, size_t inner_index) const;

  // Fetch Tensor with index from input_args_.
  Tensor *FetchTensor(size_t args_index, const KernelWithIndex &node);

  // Record graph inputs and return whether is dynamic.
  bool RecordGraphInputsAndIsDyn(const GraphCompilerInfo *graph_compiler_info, const std::vector<size_t> &input_index,
                                 const std::vector<AnfNodePtr> &parameters);

  void ConvertNormalInputContiguous(const std::vector<size_t> &input_index);

  // Release input data at the end of run graph.
  void ReleaseData();

  void SetPositionWeight(size_t outer_index, bool is_weight);
  bool GetPositionWeight(size_t outer_index);
  size_t GetNonWeightParameterNum();

  // Insert host tensor data and src device tensor into callback to avoid release before async copy finished.
  void InsertDeviceTensorIntoCallback(const DeviceAddressPtr &device_tensor);

  void SetPositionTensor(size_t outer_index, bool is_tensor);
  bool GetPositionTensor(size_t outer_index);

  void SetOffloaded(size_t outer_index, size_t inner_index, bool is_offload);
  bool GetOffloaded(size_t outer_index, size_t inner_index);

  void SetPinned(size_t outer_index, size_t inner_index, bool is_pinned);
  bool GetPinned(size_t outer_index, size_t inner_index);

  void Clear();

  const std::vector<std::vector<std::pair<KernelTensorPtr, UserCntWithPrepared>>> &GetAll() const {
    return parameter_kernel_tensors_;
  }

  const std::vector<ShapeVector> &GetHostTensorsShape() const { return host_tensors_shape_; }

  void SetParameterUsedTimes(size_t outer_index, size_t inner_index, size_t times);

  bool IsConcurrentlyUse(size_t outer_index, size_t inner_index);

  void FillBuffer(size_t idx, const std::vector<TensorPtr> &tensors);

  bool CheckBufferSize(size_t outer_index) const;
  Tensor *FlattenInputTensorByArg(size_t arg_index, const KernelWithIndex &front_node);

  std::pair<bool, std::pair<TypePtr, KernelWithIndex>> GetReleasePositionInfo(const DeviceTensorPosition &position);

  KernelTensorPtr GetReleasedCheckInfo(size_t outer_index, size_t inner_index);

 private:
  // The input args refresh in every step.
  VectorRef *input_args_;
  // The kernel tensors used for launch and transfer between actors. Outer index corresponds to the
  // front nodle position, and inner index corresponds to the addr position after tuple unfold.
  // Besides, record the user cnt and data prepared flag for each kernel tensor.
  std::vector<std::vector<std::pair<KernelTensorPtr, UserCntWithPrepared>>> parameter_kernel_tensors_;
  std::vector<bool> is_tensors_;
  std::vector<std::vector<device::DeviceType>> parameter_device_types_;
  std::vector<std::vector<bool>> is_offload_parameter_;
  std::vector<std::vector<bool>> is_parameter_pinned_;
  // Record the parameter may be concurrently used, if equal to 1, fetch parameter can not use lock.
  std::vector<std::vector<size_t>> parameter_used_times_;
  std::vector<std::vector<std::function<void(size_t)>>> async_copy_funcs_;
  // Record non-weight ref max input, so that do not transverse the store when releasing data.
  std::set<std::pair<size_t, size_t>> non_weight_ref_max_inputs_;
  std::map<DeviceTensorPosition, std::pair<TypePtr, KernelWithIndex>> release_data_info_;
  // Record released device addresses, used for check input next step.
  std::vector<std::vector<KernelTensorPtr>> released_check_addresses_;

  std::map<AnfNode *, size_t> front_node_to_index_;

  std::vector<bool> is_weights_;
  size_t weight_num_{0};
  // Subgraph non weight parameter num.
  size_t non_weight_data_num_{0};

  // When front node to index failed, use the map to find real front node.
  std::map<KernelWithIndex, KernelWithIndex> node_to_real_front_node_;
  std::map<size_t, AnfNode *> index_to_front_node_;
  // Store tensor from args.
  std::vector<std::vector<TensorPtr>> buffers_;
  size_t buffer_size_{0};
  // Protect async copy finished before release.
  std::vector<DeviceAddressPtr> device_tensor_in_callback_;
  // Record the dynamic shape for each position.
  std::vector<std::vector<bool>> is_dynamic_;
  // Record the tensor shape for inference.
  std::vector<ShapeVector> host_tensors_shape_;
  // Read/Write lock for map.
  mutable std::shared_mutex param_mutex_;
};
using GraphParameterStorePtr = std::shared_ptr<GraphParameterStore>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_CORE_GRAPH_SCHEDULER_GRAPH_PARAMETER_STORE_H_
