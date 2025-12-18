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

tensor::TensorPtr SparseFlashAttentionAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &query, const TensorPtr &key, const TensorPtr &value,
  const TensorPtr &sparse_indices, const FP32ImmPtr &scale_value, const std::optional<TensorPtr> &block_table,
  const std::optional<TensorPtr> &actual_seq_lengths_query, const std::optional<TensorPtr> &actual_seq_lengths_kv,
  const std::optional<TensorPtr> &query_rope, const std::optional<TensorPtr> &key_rope,
  const Int64ImmPtr &sparse_block_size, const Int64ImmPtr &layout_query, const Int64ImmPtr &layout_kv,
  const Int64ImmPtr &sparse_mode, const Int64ImmPtr &pre_tokens, const Int64ImmPtr &next_tokens,
  const Int64ImmPtr &attention_mode, const BoolImmPtr &return_softmax_lse) {
  OpRunner::InferOpOutput(op, query, key, value, sparse_indices, scale_value, block_table, actual_seq_lengths_query,
                          actual_seq_lengths_kv, query_rope, key_rope, sparse_block_size, layout_query, layout_kv,
                          sparse_mode, pre_tokens, next_tokens, attention_mode, return_softmax_lse);

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
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), query, key, value, sparse_indices, block_table,
                                actual_seq_lengths_query, actual_seq_lengths_kv, query_rope, key_rope);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Dispatch async run
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, query, key, value, sparse_indices, scale_value_val, block_table, actual_seq_lengths_query,
     actual_seq_lengths_kv, query_rope, key_rope, sparse_block_size_val, layout_query_val, layout_kv_val,
     sparse_mode_val, pre_tokens_val, next_tokens_val, attention_mode_val, return_softmax_lse_val]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();

      // Malloc for inputs
      PyBoostUtils::MallocOpInputs(device_context, query, key, value, sparse_indices, block_table,
                                   actual_seq_lengths_query, actual_seq_lengths_kv, query_rope, key_rope);
      // Malloc for outputs
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      MS_LOG(DEBUG) << "Run aclnnSparseFlashAttention in pyboost";

      // Convert layout strings to char*
      const char *layout_query_ptr = layout_query_val.c_str();
      const char *layout_kv_ptr = layout_kv_val.c_str();

      // Get optional tensors
      std::optional<TensorPtr> block_table_opt = block_table;
      std::optional<TensorPtr> actual_seq_lengths_query_opt = actual_seq_lengths_query;
      std::optional<TensorPtr> actual_seq_lengths_kv_opt = actual_seq_lengths_kv;
      std::optional<TensorPtr> query_rope_opt = query_rope;
      std::optional<TensorPtr> key_rope_opt = key_rope;

      LAUNCH_ACLNN(aclnnSparseFlashAttention, device_context, op->stream_id(), query, key, value, sparse_indices,
                   scale_value_val, block_table_opt, actual_seq_lengths_query_opt, actual_seq_lengths_kv_opt,
                   query_rope_opt, key_rope_opt, sparse_block_size_val, layout_query_ptr, layout_kv_ptr,
                   sparse_mode_val, pre_tokens_val, next_tokens_val, attention_mode_val, return_softmax_lse_val,
                   outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
    }));

  return op->output(0);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
