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

#include "kernel/ascend/aclnn/pyboost_impl/customize/sparse_flash_attention_grad.h"

#include <array>
#include <functional>
#include <list>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
using aclOpExecutor = device::ascend::aclOpExecutor;
using aclTensor = device::ascend::aclTensor;

namespace {
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
  static const auto acl_set_tensor_addr =
    reinterpret_cast<device::ascend::_aclSetTensorAddr>(device::ascend::GetOpApiFunc("aclSetTensorAddr"));
  if (acl_set_tensor_addr == nullptr) {
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
  acl_set_tensor_addr(executor, executor_tensor_index, tensor, address[0]);
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
    static const auto release_executor = device::ascend::OpApiDefaultResource::GetInstance().release_executor_func();
    switch (cache_type) {
      case device::ascend::ProcessCacheType::kReleaseParamsAndExecutor:
        if (release_executor != nullptr) {
          release_executor(executor_);
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
  static const auto get_workspace_size_func =
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

template <typename... Args>
void LaunchSparseFlashAttentionGrad(const device::DeviceContext *device_context, uint32_t stream_id,
                                    const Args &...args) {
  static const std::string aclnn_name = "aclnnSparseFlashAttentionGrad";
  static std::unordered_map<uint64_t, std::list<CacheTuple>::iterator> hash_map;
  static std::list<CacheTuple> hash_cache;
  static std::mutex mutex;
  static const size_t capacity = []() {
    const auto configured_capacity = device::ascend::GetCacheCapaticy();
    return configured_capacity >= 0 ? LongToSize(configured_capacity) : size_t{1024};
  }();
  static const auto simulation = common::IsCompileSimulation();
  if (simulation) {
    return;
  }

  device_context->device_res_manager_->UseStreamResInCurrentThread(stream_id);
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostLaunchAclnn,
                                     aclnn_name, false);
  auto stream_ptr = device_context->device_res_manager_->GetStream(stream_id);
  uint64_t workspace_size = 0;
  aclOpExecutor *executor = nullptr;
  std::function<void()> release_function = nullptr;
  std::function<void()> update_function = nullptr;
  {
    std::unique_lock<std::mutex> lock(mutex);
    if (capacity == 0) {
      auto [workspace, generated_executor, cache, repeat_ret] = GenerateMappedExecutor(aclnn_name, args...);
      (void)repeat_ret;
      workspace_size = workspace;
      executor = generated_executor;
      release_function = [cache]() {
        cache(device::ascend::ProcessCacheType::kReleaseParams, std::vector<std::vector<void *>>{});
      };
    } else {
      const auto hash_id = device::ascend::AclnnHash(aclnn_name, args...);
      const auto cache_iter = hash_map.find(hash_id);
      if (hash_id != 0 && cache_iter != hash_map.end()) {
        MS_LOG(DEBUG) << "SparseFlashAttentionGrad hit cache, hash id:" << hash_id;
        hash_cache.splice(hash_cache.begin(), hash_cache, cache_iter->second);
        const auto &cache_entry = hash_cache.front();
        workspace_size = std::get<3>(cache_entry);
        executor = std::get<1>(cache_entry);
        const auto cache = std::get<2>(cache_entry);
        const auto address_list = device::ascend::GetTensorAddress(args...);
        update_function = [cache, address_list]() {
          cache(device::ascend::ProcessCacheType::kUpdateTensorAddress, address_list);
        };
      } else {
        MS_LOG(INFO) << "Api " << aclnn_name << " miss cache, with hash id:" << hash_id;
        auto [workspace, generated_executor, cache, repeat_ret] = GenerateMappedExecutor(aclnn_name, args...);
        workspace_size = workspace;
        executor = generated_executor;
        if (hash_id != 0 && repeat_ret == 0) {
          hash_cache.emplace_front(hash_id, executor, cache, workspace_size);
          hash_map[hash_id] = hash_cache.begin();
          if (hash_cache.size() > capacity) {
            hash_map.erase(std::get<0>(hash_cache.back()));
            runtime::Pipeline::Get().launch_stage()->Wait();
            std::get<2>(hash_cache.back())(device::ascend::ProcessCacheType::kReleaseParamsAndExecutor, {});
            hash_cache.pop_back();
          }
        } else {
          release_function = [cache]() {
            cache(device::ascend::ProcessCacheType::kReleaseParams, std::vector<std::vector<void *>>{});
          };
        }
      }
    }
  }

  if (workspace_size == 0) {
    DISPATCH_LAUNCH_KERNEL_NO_WS(device_context, aclnn_name, executor, stream_ptr, release_function, update_function);
  } else {
    auto workspace = std::make_shared<MemBlock>(device_context, workspace_size, stream_id);
    DISPATCH_LAUNCH_KERNEL(device_context, aclnn_name, workspace->ptr_, workspace_size, executor, stream_ptr,
                           release_function, update_function);
  }
  if (runtime::RuntimeConf::GetInstance()->launch_blocking()) {
    if (!device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {
      MS_LOG(EXCEPTION) << "SyncStream failed for op " << aclnn_name;
    }
  } else {
    runtime::DeviceAddressUtils::ProcessCrossStreamAddress(aclnn_name, device_context, stream_id, args...);
  }
}
}  // namespace

SparseFlashAttentionGradResult SparseFlashAttentionGradAscendCustomize(
  const TensorPtr &query, const TensorPtr &key, const TensorPtr &value, const TensorPtr &sparse_indices,
  const TensorPtr &d_out, const TensorPtr &out, const TensorPtr &softmax_max, const TensorPtr &softmax_sum,
  const FP32ImmPtr &scale_value, const Int64ImmPtr &sparse_block_size, const std::optional<TensorPtr> &query_rope,
  const std::optional<TensorPtr> &key_rope, const std::optional<TensorPtr> &actual_seq_qlen,
  const std::optional<TensorPtr> &actual_seq_kvlen, const Int64ImmPtr &layout, const Int64ImmPtr &sparse_mode,
  const Int64ImmPtr &pre_tokens, const Int64ImmPtr &next_tokens, const Int64ImmPtr &attention_mode) {
  constexpr size_t kOutputDQuery = 0;
  constexpr size_t kOutputDKey = 1;
  constexpr size_t kOutputDValue = 2;
  constexpr size_t kOutputDQueryRope = 3;
  constexpr size_t kOutputDKeyRope = 4;
  auto outputs =
    InferOutput<5>(prim::kPrimSparseFlashAttentionGrad, query, key, value, sparse_indices, d_out, out, softmax_max,
                   softmax_sum, scale_value, sparse_block_size, query_rope, key_rope, actual_seq_qlen, actual_seq_kvlen,
                   layout, sparse_mode, pre_tokens, next_tokens, attention_mode);
  // Get scalar values
  auto scale_value_val = static_cast<double>(GetValue<float>(scale_value));
  auto sparse_block_size_val = GetValue<int64_t>(sparse_block_size);
  auto layout_val = device::ascend::FASInputLayoutMode::ConvertEnumToString(GetValue<int64_t>(layout));
  auto sparse_mode_val = GetValue<int64_t>(sparse_mode);
  auto pre_tokens_val = GetValue<int64_t>(pre_tokens);
  auto next_tokens_val = GetValue<int64_t>(next_tokens);
  const bool deterministic_val = true;

  // Prepare inputs and outputs
  TensorToDevice(query, key, value, sparse_indices, d_out, out, softmax_max, softmax_sum, query_rope, key_rope,
                 actual_seq_qlen, actual_seq_kvlen);
  TensorMalloc(outputs);

  auto device_context = CurrentDeviceContext::get();
  auto stream_id = CurrentStream::id();
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([=]() {
    LaunchSparseFlashAttentionGrad(
      device_context, stream_id, query, key, value, sparse_indices, d_out, out, softmax_max, softmax_sum,
      std::optional<TensorPtr>(actual_seq_qlen), std::optional<TensorPtr>(actual_seq_kvlen),
      std::optional<TensorPtr>(query_rope), std::optional<TensorPtr>(key_rope), scale_value_val, sparse_block_size_val,
      layout_val, sparse_mode_val, pre_tokens_val, next_tokens_val, deterministic_val, std::get<kOutputDQuery>(outputs),
      std::get<kOutputDKey>(outputs), std::get<kOutputDValue>(outputs), std::get<kOutputDQueryRope>(outputs),
      std::get<kOutputDKeyRope>(outputs));
  }));

  return outputs;
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
