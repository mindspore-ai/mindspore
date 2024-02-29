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

#include "plugin/device/ascend/kernel/pyboost/customize/flash_attention_score_grad.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "runtime/device/device_address_utils.h"
#include "transform/graph_ir/op_adapter_base.h"

namespace mindspore {
using mindspore::transform::FASInputLayoutMode;
namespace kernel {
namespace pyboost {
namespace {
void FlashAttentionScoreGradAscendCall(
  const std::shared_ptr<OpRunner> &op, const device::DeviceContext *device_context, const TensorPtr &query,
  const TensorPtr &key, const TensorPtr &value, const TensorPtr &dy, const std::optional<TensorPtr> &pse_shift,
  const std::optional<TensorPtr> &drop_mask, const std::optional<TensorPtr> &padding_mask,
  const std::optional<TensorPtr> &attn_mask, const std::optional<TensorPtr> &softmax_max,
  const std::optional<TensorPtr> &softmax_sum, const std::optional<TensorPtr> &softmax_in,
  const std::optional<TensorPtr> &attention_in, const std::optional<ValueTuplePtr> &prefix,
  const std::optional<ValueTuplePtr> &actual_seq_qlen, const std::optional<ValueTuplePtr> &actual_seq_kvlen,
  const Int64ImmPtr head_num, const FP32ImmPtr keep_prob, const FP32ImmPtr scale_value, const Int64ImmPtr pre_tokens,
  const Int64ImmPtr next_tokens, const Int64ImmPtr inner_precise, const Int64ImmPtr input_layout,
  const Int64ImmPtr sparse_mode, const std::vector<tensor::TensorPtr> &outputs) {
  std::vector<int64_t> prefix_array;
  if (prefix.has_value()) {
    prefix_array = ConvertValueTupleToVector<int64_t>(prefix.value());
  }
  auto head_num_value = GetValue<int64_t>(head_num);
  auto keep_prob_value = static_cast<double>(GetValue<float>(keep_prob));
  auto scale_value_value = static_cast<double>(GetValue<float>(scale_value));
  auto pre_tokens_value = GetValue<int64_t>(pre_tokens);
  auto next_tokens_value = GetValue<int64_t>(next_tokens);
  auto inner_precise_value = GetValue<int64_t>(inner_precise);  // not used.
  auto input_layout_string = FASInputLayoutMode::ConvertEnumToString(GetValue<int64_t>(input_layout));
  auto sparse_mode_value = GetValue<int64_t>(sparse_mode);

  if (attn_mask.has_value()) {
    auto attn_mask_tensor = attn_mask.value();
    if (attn_mask_tensor->data_type_c() == static_cast<int>(TypeId::kNumberTypeFloat16)) {
      MS_LOG(EXCEPTION) << "Attn mask don't support float16.";
    }
  }

  if (input_layout_string == "TND") {
    if (!actual_seq_kvlen.has_value() || !actual_seq_qlen.has_value()) {
      MS_LOG(EXCEPTION)
        << "For [aclnnFlashAttentionUnpaddingScoreGrad], actual_seq_qlen and actual_seq_kvlen must be not "
           "none when input layout is TND.";
    }
    std::vector<int64_t> actual_seq_qlen_array = ConvertValueTupleToVector<int64_t>(actual_seq_qlen.value());
    std::vector<int64_t> actual_seq_kvlen_array = ConvertValueTupleToVector<int64_t>(actual_seq_kvlen.value());
    LAUNCH_ACLNN(aclnnFlashAttentionUnpaddingScoreGrad, device_context, op->stream_id(), query, key, value, dy,
                 pse_shift, drop_mask, padding_mask, attn_mask, softmax_max, softmax_sum, softmax_in, attention_in,
                 prefix_array, actual_seq_qlen_array, actual_seq_kvlen_array, scale_value_value, keep_prob_value,
                 pre_tokens_value, next_tokens_value, head_num_value, input_layout_string, inner_precise_value,
                 sparse_mode_value, outputs[0], outputs[1], outputs[2], outputs[3]);
  } else {
    LAUNCH_ACLNN(aclnnFlashAttentionScoreGrad, device_context, op->stream_id(), query, key, value, dy, pse_shift,
                 drop_mask, padding_mask, attn_mask, softmax_max, softmax_sum, softmax_in, attention_in, prefix_array,
                 scale_value_value, keep_prob_value, pre_tokens_value, next_tokens_value, head_num_value,
                 input_layout_string, inner_precise_value, sparse_mode_value, outputs[0], outputs[1], outputs[2],
                 outputs[3]);
  }
}
}  // namespace

tensor::TensorPtr FlashAttentionScoreGradAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &query, const TensorPtr &key, const TensorPtr &value,
  const TensorPtr &dy, const std::optional<TensorPtr> &pse_shift, const std::optional<TensorPtr> &drop_mask,
  const std::optional<TensorPtr> &padding_mask, const std::optional<TensorPtr> &attn_mask,
  const std::optional<TensorPtr> &softmax_max, const std::optional<TensorPtr> &softmax_sum,
  const std::optional<TensorPtr> &softmax_in, const std::optional<TensorPtr> &attention_in,
  const std::optional<ValueTuplePtr> &prefix, const std::optional<ValueTuplePtr> &actual_seq_qlen,
  const std::optional<ValueTuplePtr> &actual_seq_kvlen, const Int64ImmPtr head_num, const FP32ImmPtr keep_prob,
  const FP32ImmPtr scale_value, const Int64ImmPtr pre_tokens, const Int64ImmPtr next_tokens,
  const Int64ImmPtr inner_precise, const Int64ImmPtr input_layout, const Int64ImmPtr sparse_mode) {
  OpRunner::InferOpOutput(op, query, key, value, dy, pse_shift, drop_mask, padding_mask, attn_mask, softmax_max,
                          softmax_sum, softmax_in, attention_in, prefix, actual_seq_qlen, actual_seq_kvlen, head_num,
                          keep_prob, scale_value, pre_tokens, next_tokens, inner_precise, input_layout, sparse_mode);

  // Create device address for input/output tensors
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), query, key, value, dy, pse_shift, drop_mask,
                                padding_mask, attn_mask, softmax_max, softmax_sum, softmax_in, attention_in);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, query, key, value, dy, pse_shift, drop_mask, padding_mask, attn_mask, softmax_max, softmax_sum, softmax_in,
     attention_in, prefix, actual_seq_qlen, actual_seq_kvlen, head_num, keep_prob, scale_value, pre_tokens, next_tokens,
     inner_precise, input_layout, sparse_mode]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, query, key, value, dy, pse_shift, drop_mask, padding_mask, attn_mask,
                                   softmax_max, softmax_sum, softmax_in, attention_in);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
      FlashAttentionScoreGradAscendCall(op, device_context, query, key, value, dy, pse_shift, drop_mask, padding_mask,
                                        attn_mask, softmax_max, softmax_sum, softmax_in, attention_in, prefix,
                                        actual_seq_qlen, actual_seq_kvlen, head_num, keep_prob, scale_value, pre_tokens,
                                        next_tokens, inner_precise, input_layout, sparse_mode, outputs);
      MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
