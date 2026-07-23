/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/kernel_mod_impl/customize/sparse_flash_attention_grad_aclnn_kernel.h"

#include <algorithm>
#include <array>
#include <functional>
#include <tuple>
#include <vector>
#include <string>

#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"

namespace mindspore {
namespace kernel {
namespace sparse_flash_attention_grad {

namespace {
// Input indices
constexpr size_t kInputQueryIndex = 0;
constexpr size_t kInputKeyIndex = 1;
constexpr size_t kInputValueIndex = 2;
constexpr size_t kInputSparseIndicesIndex = 3;
constexpr size_t kInputDOutIndex = 4;
constexpr size_t kInputOutIndex = 5;
constexpr size_t kInputSoftmaxMaxIndex = 6;
constexpr size_t kInputSoftmaxSumIndex = 7;
constexpr size_t kInputScaleValueIndex = 8;
constexpr size_t kInputSparseBlockSizeIndex = 9;
constexpr size_t kInputQueryRopeIndex = 10;
constexpr size_t kInputKeyRopeIndex = 11;
constexpr size_t kInputActualSeqQlenIndex = 12;
constexpr size_t kInputActualSeqKvlenIndex = 13;
constexpr size_t kInputLayoutIndex = 14;
constexpr size_t kInputSparseModeIndex = 15;
constexpr size_t kInputPreTokensIndex = 16;
constexpr size_t kInputNextTokensIndex = 17;
constexpr size_t kInputAttentionModeIndex = 18;

// Output indices
constexpr size_t kOutputDQueryIndex = 0;
constexpr size_t kOutputDKeyIndex = 1;
constexpr size_t kOutputDValueIndex = 2;
constexpr size_t kOutputDQueryRopeIndex = 3;
constexpr size_t kOutputDKeyRopeIndex = 4;

constexpr std::array<size_t, 17> kExecutorTensorIndices = {0, 1, 7, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16};

size_t GetExecutorTensorIndex(size_t public_tensor_index) {
  if (public_tensor_index >= kExecutorTensorIndices.size()) {
    MS_LOG(EXCEPTION) << "SparseFlashAttentionGrad tensor index " << public_tensor_index << " is out of range.";
  }
  return kExecutorTensorIndices[public_tensor_index];
}

void UpdateMappedAddress(aclOpExecutor *executor, aclTensor *tensor, const std::vector<void *> &address,
                         size_t *public_tensor_index) {
  MS_EXCEPTION_IF_NULL(executor);
  static const auto aclSetTensorAddr =
    reinterpret_cast<device::ascend::_aclSetTensorAddr>(device::ascend::GetOpApiFunc("aclSetTensorAddr"));
  if (aclSetTensorAddr == nullptr) {
    MS_LOG(EXCEPTION) << "aclSetTensorAddr is nullptr";
  }
  if (address.empty()) {
    return;
  }
  if (address[0] == nullptr) {
    ++(*public_tensor_index);
    return;
  }
  const auto executor_tensor_index = GetExecutorTensorIndex(*public_tensor_index);
  aclSetTensorAddr(executor, executor_tensor_index, tensor, address[0]);
  MS_LOG(DEBUG) << "SparseFlashAttentionGrad aclSetTensorAddr, public index " << *public_tensor_index
                << ", executor index " << executor_tensor_index << ", address " << address[0];
  ++(*public_tensor_index);
}

template <typename T>
void UpdateMappedAddress(aclOpExecutor *, T, const std::vector<void *> &address, size_t *public_tensor_index) {
  if (!address.empty()) {
    ++(*public_tensor_index);
  }
}

template <typename Tuple, size_t... I>
void UpdateMappedAddresses(aclOpExecutor *executor, const std::vector<std::vector<void *>> &address_list, Tuple &params,
                           std::index_sequence<I...>) {
  size_t public_tensor_index = 0;
  (void)std::initializer_list<int>{
    (UpdateMappedAddress(executor, std::get<I>(params), address_list[I], &public_tensor_index), 0)...};
}

template <typename Tuple>
void UpdateMappedAddresses(aclOpExecutor *executor, const std::vector<std::vector<void *>> &address_list,
                           Tuple &params) {
  constexpr auto kParamCount = std::tuple_size<Tuple>::value - kIndex2;
  UpdateMappedAddresses(executor, address_list, params, std::make_index_sequence<kParamCount>{});
}

template <typename T>
class SparseFlashAttentionGradCache {
 public:
  SparseFlashAttentionGradCache(aclOpExecutor *executor, T params) : executor_(executor), params_(std::move(params)) {}

  std::vector<ShapeVector> operator()(const device::ascend::ProcessCacheType &cache_type,
                                      const std::vector<std::vector<void *>> &address_list = {}) {
    static auto release_executor_func = device::ascend::OpApiDefaultResource::GetInstance().release_executor_func();
    switch (cache_type) {
      case device::ascend::ProcessCacheType::kReleaseParamsAndExecutor:
        if (release_executor_func != nullptr) {
          release_executor_func(executor_);
        }
        device::ascend::ReleaseConvertTypes(params_);
        break;
      case device::ascend::ProcessCacheType::kReleaseParams:
        device::ascend::ReleaseConvertTypes(params_);
        break;
      case device::ascend::ProcessCacheType::kUpdateTensorAddress:
        UpdateMappedAddresses(executor_, address_list, params_);
        break;
      default:
        break;
    }
    return {};
  }

 private:
  aclOpExecutor *executor_;
  T params_;
};

template <typename... Args>
auto GenerateMappedExecutor(const std::string &api_name, const Args &...args) {
  const auto workspace_api_name = api_name + "GetWorkspaceSize";
  static const auto get_workspace_size_func_ptr = device::ascend::GetOpApiFunc(workspace_api_name.c_str());
  if (get_workspace_size_func_ptr == nullptr) {
    MS_LOG(EXCEPTION) << workspace_api_name << " not in " << device::ascend::GetOpApiLibName() << ", please check!";
  }
  uint64_t workspace_size = 0;
  aclOpExecutor *executor = nullptr;
  auto converted_params = device::ascend::ConvertTypes(args..., &workspace_size, &executor);
  static auto get_workspace_size_func =
    device::ascend::ConvertToOpApiFunc(converted_params, get_workspace_size_func_ptr);
  const auto workspace_status = device::ascend::call(get_workspace_size_func, converted_params);
  if (workspace_status != 0) {
    CHECK_AND_THROW_RECOVERABLE_ERROR(workspace_api_name);
    MS_LOG(EXCEPTION) << workspace_api_name << " call failed, please check!";
  }
  const auto repeat_ret = device::ascend::SetExecutorRepeatable(workspace_api_name, executor);
  auto cache = SparseFlashAttentionGradCache<decltype(converted_params)>(executor, std::move(converted_params));
  return std::make_tuple(workspace_size, executor, device::ascend::ProcessCache(cache), repeat_ret);
}
}  // namespace

void SparseFlashAttentionGradAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs) {
  scale_value_ = static_cast<double>(device::ascend::ConvertKernelTensor<float>(inputs[kInputScaleValueIndex]));
  sparse_block_size_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kInputSparseBlockSizeIndex]);
  auto layout_int = device::ascend::ConvertKernelTensor<int64_t>(inputs[kInputLayoutIndex]);
  layout_ = device::ascend::FASInputLayoutMode::ConvertEnumToString(layout_int);
  sparse_mode_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kInputSparseModeIndex]);
  pre_tokens_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kInputPreTokensIndex]);
  next_tokens_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kInputNextTokensIndex]);
  // Reserved param; read but not passed to aclnnSparseFlashAttentionGrad (kernel does not consume it yet).
  attention_mode_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kInputAttentionModeIndex]);

  const char *layout_ptr = layout_.c_str();
  deterministic_ = true;

  auto get_workspace = [this](const auto &...args) {
    size_t workspace_size = 0;
    if (capacity_ == 0) {
      if (is_dynamic_) {
        hash_id_ = 0;
      } else {
        std::tie(workspace_size, std::ignore, std::ignore) = GEN_EXECUTOR_CUST(op_type_, args...);
      }
    } else {
      hash_id_ = device::ascend::AclnnHash(op_type_, args...);
      const auto cache_iter = hash_map_.find(hash_id_);
      if (cache_iter != hash_map_.end()) {
        hash_cache_.splice(hash_cache_.begin(), hash_cache_, cache_iter->second);
        workspace_size = std::get<3>(hash_cache_.front());
      } else {
        auto [workspace, executor, cache, repeat_ret] = GenerateMappedExecutor(op_type_, args...);
        workspace_size = workspace;
        if (repeat_ret == 0) {
          hash_cache_.emplace_front(hash_id_, executor, cache, workspace);
          hash_map_[hash_id_] = hash_cache_.begin();
          if (hash_cache_.size() > capacity_) {
            hash_map_.erase(std::get<0>(hash_cache_.back()));
            std::get<2>(hash_cache_.back())(device::ascend::ProcessCacheType::kReleaseParamsAndExecutor, {});
            hash_cache_.pop_back();
          }
        } else {
          hash_id_ = 0;
          cache(device::ascend::ProcessCacheType::kReleaseParamsAndExecutor, {});
        }
      }
    }
    if (workspace_size != 0) {
      SetWorkspaceSizeList({workspace_size});
    }
  };
  get_workspace(inputs[kInputQueryIndex], inputs[kInputKeyIndex], inputs[kInputValueIndex],
                inputs[kInputSparseIndicesIndex], inputs[kInputDOutIndex], inputs[kInputOutIndex],
                inputs[kInputSoftmaxMaxIndex], inputs[kInputSoftmaxSumIndex], inputs[kInputActualSeqQlenIndex],
                inputs[kInputActualSeqKvlenIndex], inputs[kInputQueryRopeIndex], inputs[kInputKeyRopeIndex],
                scale_value_, sparse_block_size_, layout_ptr, sparse_mode_, pre_tokens_, next_tokens_, deterministic_,
                outputs[kOutputDQueryIndex], outputs[kOutputDKeyIndex], outputs[kOutputDValueIndex],
                outputs[kOutputDQueryRopeIndex], outputs[kOutputDKeyRopeIndex]);
}

bool SparseFlashAttentionGradAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &workspace,
                                            const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  MS_LOG(DEBUG) << "Run aclnnSparseFlashAttentionGrad in kernel_mod_impl";

  const char *layout_ptr = layout_.c_str();
  auto run = [this, stream_ptr, &workspace](const auto &...args) {
    aclOpExecutor *executor = nullptr;
    std::function<void()> release_func = nullptr;
    if (capacity_ == 0) {
      size_t workspace_size = 0;
      std::tie(workspace_size, executor, release_func, hash_id_, std::ignore) =
        GEN_EXECUTOR_BOOST(op_type_, hash_id_, args...);
      if (workspace_size == 0) {
        RUN_OP_API_ASYNC(op_type_, nullptr, 0, executor, stream_ptr, release_func);
      } else if (is_dynamic_) {
        static device::DeviceContext *device_context =
          device::DeviceContextManager::GetInstance().GetDeviceContext("Ascend").get();
        auto workspace_ptr = std::make_shared<kernel::MemBlock>(device_context, workspace_size, stream_ptr);
        RUN_OP_API_ASYNC(op_type_, workspace_ptr->ptr_, workspace_size, executor, stream_ptr, release_func);
      } else {
        if (workspace.empty()) {
          MS_LOG(EXCEPTION) << "Failed to allocate workspace tensor!";
        }
        RUN_OP_API_ASYNC(op_type_, workspace[0]->device_ptr(), workspace_size, executor, stream_ptr, release_func);
      }
      return;
    }
    if (hash_map_.find(hash_id_) == hash_map_.end()) {
      std::tie(std::ignore, executor, release_func, hash_id_, std::ignore) =
        GEN_EXECUTOR_BOOST(op_type_, hash_id_, args...);
    } else {
      const auto &cache_entry = *(hash_map_.at(hash_id_));
      const auto &cache = std::get<2>(cache_entry);
      const auto address_list = device::ascend::GetTensorAddress(args...);
      cache(device::ascend::ProcessCacheType::kUpdateTensorAddress, address_list);
      executor = std::get<1>(cache_entry);
    }
    if (workspace_size_list_.empty()) {
      RUN_OP_API_ASYNC(op_type_, nullptr, 0, executor, stream_ptr, release_func);
      return;
    }
    if (workspace.empty()) {
      MS_LOG(EXCEPTION) << "Failed to allocate workspace tensor!";
    }
    const auto workspace_tensor = workspace[0];
    if (workspace_tensor->size() != workspace_size_list_[0]) {
      MS_LOG(EXCEPTION) << "Please check 'GetWorkSpaceInfo' and 'Launch' func. Expected workspace size is"
                        << workspace_size_list_[0] << ", but get " << workspace_tensor->size();
    }
    RUN_OP_API_ASYNC(op_type_, workspace_tensor->device_ptr(), workspace_size_list_[0], executor, stream_ptr,
                     release_func);
  };
  run(inputs[kInputQueryIndex], inputs[kInputKeyIndex], inputs[kInputValueIndex], inputs[kInputSparseIndicesIndex],
      inputs[kInputDOutIndex], inputs[kInputOutIndex], inputs[kInputSoftmaxMaxIndex], inputs[kInputSoftmaxSumIndex],
      inputs[kInputActualSeqQlenIndex], inputs[kInputActualSeqKvlenIndex], inputs[kInputQueryRopeIndex],
      inputs[kInputKeyRopeIndex], scale_value_, sparse_block_size_, layout_ptr, sparse_mode_, pre_tokens_, next_tokens_,
      deterministic_, outputs[kOutputDQueryIndex], outputs[kOutputDKeyIndex], outputs[kOutputDValueIndex],
      outputs[kOutputDQueryRopeIndex], outputs[kOutputDKeyRopeIndex]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(SparseFlashAttentionGrad, SparseFlashAttentionGradAscend);

}  // namespace sparse_flash_attention_grad
}  // namespace kernel
}  // namespace mindspore
