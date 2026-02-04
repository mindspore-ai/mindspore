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
#include "kernel/ascend/aclnn/pyboost_impl/customize/fused_infer_attention_score.h"
#include <utility>
#include <string>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"

namespace mindspore {
using mindspore::device::ascend::FASInputLayoutMode;
namespace kernel {
namespace pyboost {
std::vector<int64_t> FiasConvertTensorToVector(const std::string &prim_name, const std::string &tensor_name,
                                               const std::optional<tensor::TensorPtr> &tensor_opt) {
  if (!tensor_opt.has_value()) {
    return std::vector<int64_t>();
  }
  auto tensor = tensor_opt.value();
  auto tensor_cpu = tensor->cpu();
  TypeId tensor_type_id = static_cast<TypeId>(tensor_cpu->data_type_c());
  if (tensor_type_id != TypeId::kNumberTypeInt64 && tensor_type_id != TypeId::kNumberTypeInt32) {
    MS_EXCEPTION(TypeError) << "For " << prim_name << ", the input " << tensor_name
                            << " for conversion to int array must be of type Int32 or Int64,"
                            << " but got " << TypeIdToString(tensor_type_id);
  }
  std::vector<int64_t> converted_sequence;
  size_t elem_num = tensor_cpu->DataSize();
  if (tensor_type_id == TypeId::kNumberTypeInt64) {
    const int64_t *elem_ptr = static_cast<const int64_t *>(tensor_cpu->data_c());
    for (size_t i = 0; i < elem_num; i++) {
      converted_sequence.push_back(elem_ptr[i]);
    }
  } else {
    const int32_t *elem_ptr = static_cast<const int32_t *>(tensor_cpu->data_c());
    for (size_t i = 0; i < elem_num; i++) {
      converted_sequence.push_back(elem_ptr[i]);
    }
  }
  MS_LOG(DEBUG) << "Convert tensor to vector " << converted_sequence;
  return converted_sequence;
}

tensor::TensorPtr FusedInferAttentionScoreAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &query_tensor, const ValueTuplePtr &key_tensor_list,
  const ValueTuplePtr &value_tensor_list, const std::optional<TensorPtr> &pse_shift_tensor,
  const std::optional<TensorPtr> &attn_mask_tensor, const std::optional<TensorPtr> &actual_seq_lengths_tensor,
  const std::optional<TensorPtr> &actual_seq_lengths_kv_tensor, const std::optional<TensorPtr> &dequant_scale1_tensor,
  const std::optional<TensorPtr> &quant_scale1_tensor, const std::optional<TensorPtr> &dequant_scale2_tensor,
  const std::optional<TensorPtr> &quant_scale2_tensor, const std::optional<TensorPtr> &quant_offset2_tensor,
  const std::optional<TensorPtr> &antiquant_scale_tensor, const std::optional<TensorPtr> &antiquant_offset_tensor,
  const std::optional<TensorPtr> &block_table_tensor, const std::optional<TensorPtr> &query_padding_size_tensor,
  const std::optional<TensorPtr> &kv_padding_size_tensor, const std::optional<TensorPtr> &key_antiquant_scale_tensor,
  const std::optional<TensorPtr> &key_antiquant_offset_tensor,
  const std::optional<TensorPtr> &value_antiquant_scale_tensor,
  const std::optional<TensorPtr> &value_antiquant_offset_tensor,
  const std::optional<TensorPtr> &key_shared_prefix_tensor, const std::optional<TensorPtr> &value_shared_prefix_tensor,
  const std::optional<TensorPtr> &actual_shared_prefix_len_tensor, const Int64ImmPtr &num_heads,
  const FP32ImmPtr &scale_value, const Int64ImmPtr &pre_tokens, const Int64ImmPtr &next_tokens,
  const Int64ImmPtr &input_layout, const Int64ImmPtr &num_key_value_heads, const Int64ImmPtr &sparse_mode,
  const Int64ImmPtr &inner_precise, const Int64ImmPtr &block_size, const Int64ImmPtr &antiquant_mode,
  const BoolImmPtr &softmax_lse_flag, const Int64ImmPtr &key_antiquant_mode, const Int64ImmPtr &value_antiquant_mode) {
  const auto prim_name = op->primitive()->name();
  MS_LOG(DEBUG) << prim_name << " Call start";

  OpRunner::InferOpOutput(
    op, query_tensor, key_tensor_list, value_tensor_list, pse_shift_tensor, attn_mask_tensor, actual_seq_lengths_tensor,
    actual_seq_lengths_kv_tensor, dequant_scale1_tensor, quant_scale1_tensor, dequant_scale2_tensor,
    quant_scale2_tensor, quant_offset2_tensor, antiquant_scale_tensor, antiquant_offset_tensor, block_table_tensor,
    query_padding_size_tensor, kv_padding_size_tensor, key_antiquant_scale_tensor, key_antiquant_offset_tensor,
    value_antiquant_scale_tensor, value_antiquant_offset_tensor, key_shared_prefix_tensor, value_shared_prefix_tensor,
    actual_shared_prefix_len_tensor, num_heads, scale_value, pre_tokens, next_tokens, input_layout, num_key_value_heads,
    sparse_mode, inner_precise, block_size, antiquant_mode, softmax_lse_flag, key_antiquant_mode, value_antiquant_mode);

  // Convert ValueTuple to std::vector
  std::vector<TensorPtr> key_tensor_list_vector = ConvertValueTupleToVector<TensorPtr>(key_tensor_list);
  std::vector<TensorPtr> value_tensor_list_vector = ConvertValueTupleToVector<TensorPtr>(value_tensor_list);

  // Convert Tensor to std::vector
  auto actual_seq_lengths_vector =
    FiasConvertTensorToVector(prim_name, "actual_seq_lengths", actual_seq_lengths_tensor);
  auto actual_seq_lengths_vector_pair = std::make_pair(actual_seq_lengths_vector, true);
  auto actual_seq_lengths_kv_vector =
    FiasConvertTensorToVector(prim_name, "actual_seq_lengths_kv", actual_seq_lengths_kv_tensor);
  auto actual_seq_lengths_kv_vector_pair = std::make_pair(actual_seq_lengths_kv_vector, true);
  auto actual_shared_prefix_len_vector =
    FiasConvertTensorToVector(prim_name, "actual_shared_prefix_len", actual_shared_prefix_len_tensor);
  auto actual_shared_prefix_len_vector_pair = std::make_pair(actual_shared_prefix_len_vector, true);

  // Convert ValuePtr to c++ scalar
  auto num_heads_imm = GetValue<int64_t>(num_heads);
  auto scale_value_imm = GetValue<float>(scale_value);
  // Note: aclnn requires the scale_value to be of type double. If a float value is passed, aclnn will internally
  // treat it as 0, which will result in incorrect computation.
  double scale_value_imm_d = static_cast<double>(scale_value_imm);
  auto pre_tokens_imm = GetValue<int64_t>(pre_tokens);
  auto next_tokens_imm = GetValue<int64_t>(next_tokens);
  auto input_layout_imm = FASInputLayoutMode::ConvertEnumToString(GetValue<int64_t>(input_layout));
  auto num_key_value_heads_imm = GetValue<int64_t>(num_key_value_heads);
  auto sparse_mode_imm = GetValue<int64_t>(sparse_mode);
  auto inner_precise_imm = GetValue<int64_t>(inner_precise);
  auto block_size_imm = GetValue<int64_t>(block_size);
  auto antiquant_mode_imm = GetValue<int64_t>(antiquant_mode);
  auto softmax_lse_flag_imm = GetValue<bool>(softmax_lse_flag);
  auto key_antiquant_mode_imm = GetValue<int64_t>(key_antiquant_mode);
  auto value_antiquant_mode_imm = GetValue<int64_t>(value_antiquant_mode);

  PyBoostUtils::PrepareOpInputs(
    op->device_context(), op->stream_id(), query_tensor, key_tensor_list_vector, value_tensor_list_vector,
    pse_shift_tensor, attn_mask_tensor, dequant_scale1_tensor, quant_scale1_tensor, dequant_scale2_tensor,
    quant_scale2_tensor, quant_offset2_tensor, antiquant_scale_tensor, antiquant_offset_tensor, block_table_tensor,
    query_padding_size_tensor, kv_padding_size_tensor, key_antiquant_scale_tensor, key_antiquant_offset_tensor,
    value_antiquant_scale_tensor, value_antiquant_offset_tensor, key_shared_prefix_tensor, value_shared_prefix_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [prim_name, op, query_tensor, key_tensor_list_vector, value_tensor_list_vector, pse_shift_tensor, attn_mask_tensor,
     actual_seq_lengths_vector_pair, actual_seq_lengths_kv_vector_pair, dequant_scale1_tensor, quant_scale1_tensor,
     dequant_scale2_tensor, quant_scale2_tensor, quant_offset2_tensor, antiquant_scale_tensor, antiquant_offset_tensor,
     block_table_tensor, query_padding_size_tensor, kv_padding_size_tensor, key_antiquant_scale_tensor,
     key_antiquant_offset_tensor, value_antiquant_scale_tensor, value_antiquant_offset_tensor, key_shared_prefix_tensor,
     value_shared_prefix_tensor, actual_shared_prefix_len_vector_pair, num_heads_imm, scale_value_imm_d, pre_tokens_imm,
     next_tokens_imm, input_layout_imm, num_key_value_heads_imm, sparse_mode_imm, inner_precise_imm, block_size_imm,
     antiquant_mode_imm, softmax_lse_flag_imm, key_antiquant_mode_imm, value_antiquant_mode_imm]() {
      MS_LOG(DEBUG) << "Run device task " << prim_name << " start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, query_tensor, key_tensor_list_vector, value_tensor_list_vector,
                                   pse_shift_tensor, attn_mask_tensor, dequant_scale1_tensor, quant_scale1_tensor,
                                   dequant_scale2_tensor, quant_scale2_tensor, quant_offset2_tensor,
                                   antiquant_scale_tensor, antiquant_offset_tensor, block_table_tensor,
                                   query_padding_size_tensor, kv_padding_size_tensor, key_antiquant_scale_tensor,
                                   key_antiquant_offset_tensor, value_antiquant_scale_tensor,
                                   value_antiquant_offset_tensor, key_shared_prefix_tensor, value_shared_prefix_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnFusedInferAttentionScoreV2, device_context, op->stream_id(), query_tensor,
                   key_tensor_list_vector, value_tensor_list_vector, pse_shift_tensor, attn_mask_tensor,
                   actual_seq_lengths_vector_pair, actual_seq_lengths_kv_vector_pair, dequant_scale1_tensor,
                   quant_scale1_tensor, dequant_scale2_tensor, quant_scale2_tensor, quant_offset2_tensor,
                   antiquant_scale_tensor, antiquant_offset_tensor, block_table_tensor, query_padding_size_tensor,
                   kv_padding_size_tensor, key_antiquant_scale_tensor, key_antiquant_offset_tensor,
                   value_antiquant_scale_tensor, value_antiquant_offset_tensor, key_shared_prefix_tensor,
                   value_shared_prefix_tensor, actual_shared_prefix_len_vector_pair, num_heads_imm, scale_value_imm_d,
                   pre_tokens_imm, next_tokens_imm, input_layout_imm, num_key_value_heads_imm, sparse_mode_imm,
                   inner_precise_imm, block_size_imm, antiquant_mode_imm, softmax_lse_flag_imm, key_antiquant_mode_imm,
                   value_antiquant_mode_imm, outputs[0], outputs[1]);
      MS_LOG(DEBUG) << "Run device task " << prim_name << " end";
    }));
  MS_LOG(DEBUG) << prim_name << " Call end";
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
