/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/prompt_flash_attention.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"

namespace mindspore {
using mindspore::device::ascend::FASInputLayoutMode;
namespace kernel {
namespace pyboost {
namespace {

void PromptFlashAttentionAscendCall(
  const std::shared_ptr<OpRunner> &op, const device::DeviceContext *device_context, const TensorPtr &query,
  const TensorPtr &key, const TensorPtr &value, const std::optional<TensorPtr> &atten_mask,
  const std::optional<ValueTuplePtr> &actual_seq_qlen, const std::optional<ValueTuplePtr> &actual_seq_qlen_kv,
  const std::optional<TensorPtr> &pse_shift, const std::optional<TensorPtr> &deq_scale1,
  const std::optional<TensorPtr> &quant_scale1, const std::optional<TensorPtr> &deq_scale2,
  const std::optional<TensorPtr> &quant_scale2, const std::optional<TensorPtr> &quant_offset2,
  const Int64ImmPtr num_heads, const FP32ImmPtr scale_value, const Int64ImmPtr pre_tokens,
  const Int64ImmPtr next_tokens, const Int64ImmPtr input_layout, const Int64ImmPtr num_key_value_heads,
  const Int64ImmPtr sparse_mode, const Int64ImmPtr inner_precise, const std::vector<tensor::TensorPtr> &outputs) {
  std::vector<int64_t> actual_seq_qlen_array;
  std::vector<int64_t> actual_seq_qlen_kv_array;

  auto num_heads_value = GetValue<int64_t>(num_heads);
  auto scale_value_value = static_cast<double>(GetValue<float>(scale_value));
  auto pre_tokens_value = GetValue<int64_t>(pre_tokens);
  auto next_tokens_value = GetValue<int64_t>(next_tokens);
  auto input_layout_string = FASInputLayoutMode::ConvertEnumToString(GetValue<int64_t>(input_layout));
  auto num_key_value_heads_value = GetValue<int64_t>(num_key_value_heads);
  auto sparse_mode_value = GetValue<int64_t>(sparse_mode);
  auto inner_precise_value = GetValue<int64_t>(inner_precise);

  if (actual_seq_qlen.has_value()) {
    actual_seq_qlen_array = ConvertValueTupleToVector<int64_t>(actual_seq_qlen.value());
  }
  if (actual_seq_qlen_kv.has_value()) {
    actual_seq_qlen_kv_array = ConvertValueTupleToVector<int64_t>(actual_seq_qlen_kv.value());
  }
  LAUNCH_ACLNN(aclnnPromptFlashAttentionV3, device_context, op->stream_id(), query, key, value, pse_shift, atten_mask,
               actual_seq_qlen_array, actual_seq_qlen_kv_array, deq_scale1, quant_scale1, deq_scale2, quant_scale2,
               quant_offset2, num_heads_value, scale_value_value, pre_tokens_value, next_tokens_value,
               input_layout_string, num_key_value_heads_value, sparse_mode_value, inner_precise_value, outputs[0]);
}
}  // namespace

tensor::TensorPtr PromptFlashAttentionAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &query, const TensorPtr &key, const TensorPtr &value,
  const std::optional<TensorPtr> &atten_mask, const std::optional<ValueTuplePtr> &actual_seq_qlen,
  const std::optional<ValueTuplePtr> &actual_seq_qlen_kv, const std::optional<TensorPtr> &pse_shift,
  const std::optional<TensorPtr> &deq_scale1, const std::optional<TensorPtr> &quant_scale1,
  const std::optional<TensorPtr> &deq_scale2, const std::optional<TensorPtr> &quant_scale2,
  const std::optional<TensorPtr> &quant_offset2, const Int64ImmPtr num_heads, const FP32ImmPtr scale_value,
  const Int64ImmPtr pre_tokens, const Int64ImmPtr next_tokens, const Int64ImmPtr input_layout,
  const Int64ImmPtr num_key_value_heads, const Int64ImmPtr sparse_mode, const Int64ImmPtr inner_precise) {
  OpRunner::InferOpOutput(op, query, key, value, atten_mask, actual_seq_qlen, actual_seq_qlen_kv, pse_shift, deq_scale1,
                          quant_scale1, deq_scale2, quant_scale2, quant_offset2, num_heads, scale_value, pre_tokens,
                          next_tokens, input_layout, num_key_value_heads, sparse_mode, inner_precise);

  // Create device address for input/output tensors
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), query, key, value, atten_mask, pse_shift,
                                deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, query, key, value, atten_mask, actual_seq_qlen, actual_seq_qlen_kv, pse_shift, deq_scale1, quant_scale1,
     deq_scale2, quant_scale2, quant_offset2, num_heads, scale_value, pre_tokens, next_tokens, input_layout,
     num_key_value_heads, sparse_mode, inner_precise]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, query, key, value, atten_mask, pse_shift, deq_scale1, quant_scale1,
                                   deq_scale2, quant_scale2, quant_offset2);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
      PromptFlashAttentionAscendCall(op, device_context, query, key, value, atten_mask, actual_seq_qlen,
                                     actual_seq_qlen_kv, pse_shift, deq_scale1, quant_scale1, deq_scale2, quant_scale2,
                                     quant_offset2, num_heads, scale_value, pre_tokens, next_tokens, input_layout,
                                     num_key_value_heads, sparse_mode, inner_precise, outputs);
      MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
