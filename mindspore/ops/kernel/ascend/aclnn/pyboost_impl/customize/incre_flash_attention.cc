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

#include "kernel/ascend/aclnn/pyboost_impl/customize/incre_flash_attention.h"
#include <memory>
#include <string>
#include <utility>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

std::vector<int64_t> ConvertActualSeqLengthsToVector(const std::optional<tensor::TensorPtr> &tensor_opt) {
  if (!tensor_opt.has_value()) {
    return std::vector<int64_t>();
  }
  auto tensor = tensor_opt.value();
  auto tensor_cpu = tensor->cpu();
  TypeId tensor_type_id = static_cast<TypeId>(tensor_cpu->data_type_c());
  if (tensor_type_id != TypeId::kNumberTypeInt64 && tensor_type_id != TypeId::kNumberTypeInt32) {
    MS_LOG(EXCEPTION) << "Data type of actual seq length must be Int64 or Int32, "
                      << "but get " << TypeIdToString(tensor_type_id);
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

tensor::TensorPtr IncreFlashAttentionAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &query_tensor, const ValueTuplePtr &key_tensor_list,
  const ValueTuplePtr &value_tensor_list, const std::optional<TensorPtr> &attn_mask_tensor,
  const std::optional<TensorPtr> &actual_seq_lengths_tensor, const std::optional<TensorPtr> &pse_shift_tensor,
  const std::optional<TensorPtr> &dequant_scale1_tensor, const std::optional<TensorPtr> &quant_scale1_tensor,
  const std::optional<TensorPtr> &dequant_scale2_tensor, const std::optional<TensorPtr> &quant_scale2_tensor,
  const std::optional<TensorPtr> &quant_offset2_tensor, const std::optional<TensorPtr> &antiquant_scale_tensor,
  const std::optional<TensorPtr> &antiquant_offset_tensor, const std::optional<TensorPtr> &block_table_tensor,
  const std::optional<TensorPtr> &kv_padding_size_tensor, const Int64ImmPtr &num_heads, const Int64ImmPtr &input_layout,
  const FP32ImmPtr &scale_value, const Int64ImmPtr &num_key_value_heads, const Int64ImmPtr &block_size,
  const Int64ImmPtr &inner_precise) {
  OpRunner::InferOpOutput(op, query_tensor, key_tensor_list, value_tensor_list, attn_mask_tensor,
                          actual_seq_lengths_tensor, pse_shift_tensor, dequant_scale1_tensor, quant_scale1_tensor,
                          dequant_scale2_tensor, quant_scale2_tensor, quant_offset2_tensor, antiquant_scale_tensor,
                          antiquant_offset_tensor, block_table_tensor, kv_padding_size_tensor, num_heads, input_layout,
                          scale_value, num_key_value_heads, block_size, inner_precise);
  // ValueTuple to std::vector
  // ValueTuple to std::vector
  std::vector<TensorPtr> key_tensor_list_vector = ConvertValueTupleToVector<TensorPtr>(key_tensor_list);
  std::vector<TensorPtr> value_tensor_list_vector = ConvertValueTupleToVector<TensorPtr>(value_tensor_list);
  auto actual_seq_lengths_vector = ConvertActualSeqLengthsToVector(actual_seq_lengths_tensor);
  auto actual_seq_lengths_vector_pair = std::make_pair(actual_seq_lengths_vector, true);
  // Convert ValuePtr to c++ scalar
  // Convert ValuePtr to c++ scalar
  auto num_heads_imm = GetValue<int64_t>(num_heads);
  auto input_layout_imm = GetValue<int64_t>(input_layout);
  auto input_layout_str = device::ascend::FASInputLayoutMode::ConvertEnumToString(input_layout_imm);
  auto scale_value_imm = GetValue<float>(scale_value);
  double scale_value_imm_d = static_cast<double>(scale_value_imm);
  auto num_key_value_heads_imm = GetValue<int64_t>(num_key_value_heads);
  auto block_size_imm = GetValue<int64_t>(block_size);
  auto inner_precise_imm = GetValue<int64_t>(inner_precise);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), query_tensor, key_tensor_list_vector,
                                value_tensor_list_vector, pse_shift_tensor, attn_mask_tensor, dequant_scale1_tensor,
                                quant_scale1_tensor, dequant_scale2_tensor, quant_scale2_tensor, quant_offset2_tensor,
                                antiquant_scale_tensor, antiquant_offset_tensor, block_table_tensor,
                                kv_padding_size_tensor);

  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, query_tensor, key_tensor_list_vector, value_tensor_list_vector, pse_shift_tensor, attn_mask_tensor,
     actual_seq_lengths_vector_pair, dequant_scale1_tensor, quant_scale1_tensor, dequant_scale2_tensor,
     quant_scale2_tensor, quant_offset2_tensor, antiquant_scale_tensor, antiquant_offset_tensor, block_table_tensor,
     kv_padding_size_tensor, num_heads_imm, scale_value_imm_d, input_layout_str, num_key_value_heads_imm,
     block_size_imm, inner_precise_imm]() {
      const string op_name = "IncreFlashAttention";
      MS_LOG(DEBUG) << "Run device task " << op_name << " start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, query_tensor, key_tensor_list_vector, value_tensor_list_vector,
                                   pse_shift_tensor, attn_mask_tensor, dequant_scale1_tensor, quant_scale1_tensor,
                                   dequant_scale2_tensor, quant_scale2_tensor, quant_offset2_tensor,
                                   antiquant_scale_tensor, antiquant_offset_tensor, block_table_tensor,
                                   kv_padding_size_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnIncreFlashAttentionV4, device_context, op->stream_id(), query_tensor, key_tensor_list_vector,
                   value_tensor_list_vector, pse_shift_tensor, attn_mask_tensor, actual_seq_lengths_vector_pair,
                   dequant_scale1_tensor, quant_scale1_tensor, dequant_scale2_tensor, quant_scale2_tensor,
                   quant_offset2_tensor, antiquant_scale_tensor, antiquant_offset_tensor, block_table_tensor,
                   kv_padding_size_tensor, num_heads_imm, scale_value_imm_d, input_layout_str, num_key_value_heads_imm,
                   block_size_imm, inner_precise_imm, outputs[0]);

      MS_LOG(DEBUG) << "Run device task " << op_name << " end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
