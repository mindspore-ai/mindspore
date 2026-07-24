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

#include <functional>
#include <memory>
#include <string>
#include <tuple>

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
void LaunchSparseFlashAttentionGrad(const device::DeviceContext *device_context, uint32_t stream_id,
                                    const Args &...args) {
  static const std::string aclnn_name = "aclnnSparseFlashAttentionGrad";
  static const auto simulation = common::IsCompileSimulation();
  if (simulation) {
    return;
  }

  device_context->device_res_manager_->UseStreamResInCurrentThread(stream_id);
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostLaunchAclnn,
                                     aclnn_name, false);
  auto stream_ptr = device_context->device_res_manager_->GetStream(stream_id);
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
