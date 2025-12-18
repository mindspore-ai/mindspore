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

#include <memory>
#include <string>
#include <vector>

#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

std::vector<tensor::TensorPtr> SparseFlashAttentionGradAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &query, const TensorPtr &key, const TensorPtr &value,
  const TensorPtr &sparse_indices, const TensorPtr &d_out, const TensorPtr &out, const TensorPtr &softmax_max,
  const TensorPtr &softmax_sum, const FP32ImmPtr &scale_value, const Int64ImmPtr &sparse_block_size,
  const std::optional<TensorPtr> &query_rope, const std::optional<TensorPtr> &key_rope,
  const std::optional<TensorPtr> &actual_seq_qlen, const std::optional<TensorPtr> &actual_seq_kvlen,
  const Int64ImmPtr &layout, const Int64ImmPtr &sparse_mode, const Int64ImmPtr &pre_tokens,
  const Int64ImmPtr &next_tokens) {
  OpRunner::InferOpOutput(op, query, key, value, sparse_indices, d_out, out, softmax_max, softmax_sum, scale_value,
                          sparse_block_size, query_rope, key_rope, actual_seq_qlen, actual_seq_kvlen, layout,
                          sparse_mode, pre_tokens, next_tokens);

  // Get scalar values
  auto scale_value_val = static_cast<double>(GetValue<float>(scale_value));
  auto sparse_block_size_val = GetValue<int64_t>(sparse_block_size);
  auto layout_val = device::ascend::FASInputLayoutMode::ConvertEnumToString(GetValue<int64_t>(layout));
  auto sparse_mode_val = GetValue<int64_t>(sparse_mode);
  auto pre_tokens_val = GetValue<int64_t>(pre_tokens);
  auto next_tokens_val = GetValue<int64_t>(next_tokens);
  const bool deterministic_val = true;

  // Prepare inputs and outputs
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), query, key, value, sparse_indices, d_out, out,
                                softmax_max, softmax_sum, query_rope, key_rope, actual_seq_qlen, actual_seq_kvlen);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Dispatch async run
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, query, key, value, sparse_indices, d_out, out, softmax_max, softmax_sum, scale_value_val,
     sparse_block_size_val, query_rope, key_rope, actual_seq_qlen, actual_seq_kvlen, layout_val, sparse_mode_val,
     pre_tokens_val, next_tokens_val, deterministic_val]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();

      // Malloc for inputs
      PyBoostUtils::MallocOpInputs(device_context, query, key, value, sparse_indices, d_out, out, softmax_max,
                                   softmax_sum, query_rope, key_rope, actual_seq_qlen, actual_seq_kvlen);
      // Malloc for outputs
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      MS_LOG(DEBUG) << "Run aclnnSparseFlashAttentionGrad in pyboost";

      // Convert layout string to char*
      const char *layout_ptr = layout_val.c_str();

      // Get optional tensors
      std::optional<TensorPtr> actual_seq_qlen_opt = actual_seq_qlen;
      std::optional<TensorPtr> actual_seq_kvlen_opt = actual_seq_kvlen;
      std::optional<TensorPtr> query_rope_opt = query_rope;
      std::optional<TensorPtr> key_rope_opt = key_rope;

      LAUNCH_ACLNN(aclnnSparseFlashAttentionGrad, device_context, op->stream_id(), query, key, value, sparse_indices,
                   d_out, out, softmax_max, softmax_sum, actual_seq_qlen_opt, actual_seq_kvlen_opt, query_rope_opt,
                   key_rope_opt, scale_value_val, sparse_block_size_val, layout_ptr, sparse_mode_val, pre_tokens_val,
                   next_tokens_val, deterministic_val, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2],
                   outputs[kIndex3], outputs[kIndex4]);
    }));

  return op->outputs();
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
