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

#include "plugin/device/ascend/kernel/pyboost/customize/fused_infer_attention_score.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "runtime/device/device_address_utils.h"
#include "transform/graph_ir/op_adapter_base.h"

namespace mindspore {
using mindspore::transform::FASInputLayoutMode;
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr FusedInferAttentionScoreAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &query_tensor, const ValueTuplePtr &key_tensor_list,
  const ValueTuplePtr &value_tensor_list, const std::optional<BaseTensorPtr> &pse_shift_tensor,
  const std::optional<BaseTensorPtr> &attn_mask_tensor, const std::optional<ValueTuplePtr> &actual_seq_lengths,
  const std::optional<ValueTuplePtr> &actual_seq_lengths_kv, const std::optional<BaseTensorPtr> &dequant_scale1_tensor,
  const std::optional<BaseTensorPtr> &quant_scale1_tensor, const std::optional<BaseTensorPtr> &dequant_scale2_tensor,
  const std::optional<BaseTensorPtr> &quant_scale2_tensor, const std::optional<BaseTensorPtr> &quant_offset2_tensor,
  const std::optional<BaseTensorPtr> &antiquant_scale_tensor,
  const std::optional<BaseTensorPtr> &antiquant_offset_tensor, const std::optional<BaseTensorPtr> &block_table_tensor,
  const std::optional<BaseTensorPtr> &query_padding_size_tensor,
  const std::optional<BaseTensorPtr> &kv_padding_size_tensor, const Int64ImmPtr &num_heads, const FP32ImmPtr &scale,
  const Int64ImmPtr &pre_tokens, const Int64ImmPtr &next_tokens, const Int64ImmPtr &input_layout,
  const Int64ImmPtr &num_key_value_heads, const Int64ImmPtr &sparse_mode, const Int64ImmPtr &inner_precise,
  const Int64ImmPtr &block_size, const Int64ImmPtr &antiquant_mode, const BoolImmPtr &softmax_lse_flag) {
  // infer
  OpRunner::InferOpOutput(
    op, query_tensor, key_tensor_list, value_tensor_list, pse_shift_tensor, attn_mask_tensor, actual_seq_lengths,
    actual_seq_lengths_kv, dequant_scale1_tensor, quant_scale1_tensor, dequant_scale2_tensor, quant_scale2_tensor,
    quant_offset2_tensor, antiquant_scale_tensor, antiquant_offset_tensor, block_table_tensor,
    query_padding_size_tensor, kv_padding_size_tensor, num_heads, scale, pre_tokens, next_tokens, input_layout,
    num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, softmax_lse_flag);
  // ValueTuple to std::vector
  std::vector<BaseTensorPtr> key_tensor_list_vector = ConvertValueTupleToVector<BaseTensorPtr>(key_tensor_list);
  std::vector<BaseTensorPtr> value_tensor_list_vector = ConvertValueTupleToVector<BaseTensorPtr>(value_tensor_list);
  std::vector<int64_t> actual_seq_lengths_vector;
  std::vector<int64_t> actual_seq_lengths_kv_vector;
  if (actual_seq_lengths.has_value()) {
    actual_seq_lengths_vector = ConvertValueTupleToVector<int64_t>(actual_seq_lengths.value());
  }
  if (actual_seq_lengths_kv.has_value()) {
    actual_seq_lengths_kv_vector = ConvertValueTupleToVector<int64_t>(actual_seq_lengths_kv.value());
  }
  // Convert ValuePtr to c++ scalar
  auto num_heads_imm = GetValue<int64_t>(num_heads);
  auto scale_imm = static_cast<double>(GetValue<float>(scale));
  auto pre_tokens_imm = GetValue<int64_t>(pre_tokens);
  auto next_tokens_imm = GetValue<int64_t>(next_tokens);
  auto input_layout_imm = FASInputLayoutMode::ConvertEnumToString(GetValue<int64_t>(input_layout));
  auto num_key_value_heads_imm = GetValue<int64_t>(num_key_value_heads);
  auto sparse_mode_imm = GetValue<int64_t>(sparse_mode);
  auto inner_precise_imm = GetValue<int64_t>(inner_precise);
  auto block_size_imm = GetValue<int64_t>(block_size);
  auto antiquant_mode_imm = GetValue<int64_t>(antiquant_mode);
  auto softmax_lse_flag_imm = GetValue<bool>(softmax_lse_flag);
  // Create device address for input/output tensors
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), query_tensor, key_tensor_list_vector,
                                value_tensor_list_vector, pse_shift_tensor, attn_mask_tensor, dequant_scale1_tensor,
                                quant_scale1_tensor, dequant_scale2_tensor, quant_scale2_tensor, quant_offset2_tensor,
                                antiquant_scale_tensor, antiquant_offset_tensor, block_table_tensor,
                                query_padding_size_tensor, kv_padding_size_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, query_tensor, key_tensor_list_vector, value_tensor_list_vector, pse_shift_tensor, attn_mask_tensor,
     actual_seq_lengths_vector, actual_seq_lengths_kv_vector, dequant_scale1_tensor, quant_scale1_tensor,
     dequant_scale2_tensor, quant_scale2_tensor, quant_offset2_tensor, antiquant_scale_tensor, antiquant_offset_tensor,
     block_table_tensor, query_padding_size_tensor, kv_padding_size_tensor, num_heads_imm, scale_imm, pre_tokens_imm,
     next_tokens_imm, input_layout_imm, num_key_value_heads_imm, sparse_mode_imm, inner_precise_imm, block_size_imm,
     antiquant_mode_imm, softmax_lse_flag_imm]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, query_tensor, key_tensor_list_vector, value_tensor_list_vector,
                                   pse_shift_tensor, attn_mask_tensor, dequant_scale1_tensor, quant_scale1_tensor,
                                   dequant_scale2_tensor, quant_scale2_tensor, quant_offset2_tensor,
                                   antiquant_scale_tensor, antiquant_offset_tensor, block_table_tensor,
                                   query_padding_size_tensor, kv_padding_size_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
      LAUNCH_ACLNN(aclnnFusedInferAttentionScore, device_context, op->stream_id(), query_tensor, key_tensor_list_vector,
                   value_tensor_list_vector, pse_shift_tensor, attn_mask_tensor, actual_seq_lengths_vector,
                   actual_seq_lengths_kv_vector, dequant_scale1_tensor, quant_scale1_tensor, dequant_scale2_tensor,
                   quant_scale2_tensor, quant_offset2_tensor, antiquant_scale_tensor, antiquant_offset_tensor,
                   block_table_tensor, query_padding_size_tensor, kv_padding_size_tensor, num_heads_imm, scale_imm,
                   pre_tokens_imm, next_tokens_imm, input_layout_imm, num_key_value_heads_imm, sparse_mode_imm,
                   inner_precise_imm, block_size_imm, antiquant_mode_imm, softmax_lse_flag_imm, outputs[0], outputs[1]);
      MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
