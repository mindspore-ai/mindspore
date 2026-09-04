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

#include "kernel/ascend/aclnn/pyboost_impl/customize/sparse_flash_attention.h"

#include <functional>
#include <memory>
#include <string>

#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
template <typename... Args>
void LaunchSparseFlashAttention(const device::DeviceContext *device_context, uint32_t stream_id, const Args &...args) {
  static const std::string aclnn_name = "aclnnSparseFlashAttention";
  static const auto simulation = common::IsCompileSimulation();
  if (simulation) {
    return;
  }

  device_context->device_res_manager_->UseStreamResInCurrentThread(stream_id);
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostLaunchAclnn,
                                     aclnn_name, false);
  auto stream_ptr = device_context->device_res_manager_->GetStream(stream_id);
  // Do not keep a MindSpore-side executor cache and do not refresh tensor addresses while
  // reusing an executor. Address refresh (aclSetTensorAddr) indexes the executor's internal
  // tensor slots sequentially, which depends on the CANN tensor layout and breaks when a newer
  // CANN inserts an input (e.g. the optional `sinks`) that MindSpore does not expose.
  // GEN_EXECUTOR may still hit a CANN-internal executor cache when the CANN package provides one,
  // but since MindSpore never updates addresses against a cached executor here, correctness never
  // depends on the executor's internal tensor layout.
  auto executor_info = GEN_EXECUTOR(aclnn_name, args...);
  auto workspace_size = std::get<0>(executor_info);
  auto executor = std::get<1>(executor_info);
  auto release_function = std::get<3>(executor_info);
  std::function<void()> update_function = nullptr;
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

std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> SparseFlashAttentionAscendCustomize(
  const TensorPtr &query, const TensorPtr &key, const TensorPtr &value, const TensorPtr &sparse_indices,
  const FP32ImmPtr &scale_value, const std::optional<TensorPtr> &block_table,
  const std::optional<TensorPtr> &actual_seq_lengths_query, const std::optional<TensorPtr> &actual_seq_lengths_kv,
  const std::optional<TensorPtr> &query_rope, const std::optional<TensorPtr> &key_rope,
  const Int64ImmPtr &sparse_block_size, const Int64ImmPtr &layout_query, const Int64ImmPtr &layout_kv,
  const Int64ImmPtr &sparse_mode, const Int64ImmPtr &pre_tokens, const Int64ImmPtr &next_tokens,
  const Int64ImmPtr &attention_mode, const BoolImmPtr &return_softmax_lse) {
  constexpr size_t kOutputAttentionOut = 0;
  constexpr size_t kOutputSoftmaxMax = 1;
  constexpr size_t kOutputSoftmaxSum = 2;
  MS_LOG(DEBUG) << "Call aclnnSparseFlashAttention start";
  auto outputs =
    InferOutput<3>(prim::kPrimSparseFlashAttention, query, key, value, sparse_indices, scale_value, block_table,
                   actual_seq_lengths_query, actual_seq_lengths_kv, query_rope, key_rope, sparse_block_size,
                   layout_query, layout_kv, sparse_mode, pre_tokens, next_tokens, attention_mode, return_softmax_lse);

  // Get scalar values
  auto scale_value_val = static_cast<double>(GetValue<float>(scale_value));
  auto sparse_block_size_val = GetValue<int64_t>(sparse_block_size);
  auto layout_query_val = device::ascend::FASInputLayoutMode::ConvertEnumToString(GetValue<int64_t>(layout_query));
  auto layout_kv_val = device::ascend::FASInputLayoutMode::ConvertEnumToString(GetValue<int64_t>(layout_kv));
  auto sparse_mode_val = GetValue<int64_t>(sparse_mode);
  auto pre_tokens_val = GetValue<int64_t>(pre_tokens);
  auto next_tokens_val = GetValue<int64_t>(next_tokens);
  auto attention_mode_val = GetValue<int64_t>(attention_mode);
  auto return_softmax_lse_val = GetValue<bool>(return_softmax_lse);

  // Prepare inputs and outputs
  TensorToDevice(query, key, value, sparse_indices, block_table, actual_seq_lengths_query, actual_seq_lengths_kv,
                 query_rope, key_rope);
  TensorMalloc(outputs);

  auto device_context = CurrentDeviceContext::get();
  auto stream_id = CurrentStream::id();
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([=]() {
    LaunchSparseFlashAttention(device_context, stream_id, query, key, value, sparse_indices, scale_value_val,
                               std::optional<TensorPtr>(block_table),
                               std::optional<TensorPtr>(actual_seq_lengths_query),
                               std::optional<TensorPtr>(actual_seq_lengths_kv), std::optional<TensorPtr>(query_rope),
                               std::optional<TensorPtr>(key_rope), sparse_block_size_val, layout_query_val.c_str(),
                               layout_kv_val.c_str(), sparse_mode_val, pre_tokens_val, next_tokens_val,
                               attention_mode_val, return_softmax_lse_val, std::get<kOutputAttentionOut>(outputs),
                               std::get<kOutputSoftmaxMax>(outputs), std::get<kOutputSoftmaxSum>(outputs));
  }));

  MS_LOG(DEBUG) << "Launch aclnnSparseFlashAttention end";
  return outputs;
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
