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

#include "kernel/ascend/aclnn/pyboost_impl/customize/speed_fusion_attention.h"
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/dropout_gen_mask_ext.h"
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/zeros.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
#include "include/op_enum.h"
#include "ir/tensor_new.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
std::optional<TensorPtr> SpeedFusionAttentionDropoutGenMaskCall(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &query, const TensorPtr &key, int64_t head_num_value,
  int64_t input_layout_value, double keep_prob_value, const TensorPtr &seed, const TensorPtr &offset,
  const BoolImmPtr &gen_mask_parallel, const BoolImmPtr &sync, int64_t *numels) {
  auto query_shape = query->shape();
  auto key_shape = key->shape();

  switch (input_layout_value) {
    case mindspore::ops::FASInputLayoutMode::BSH:
      *numels = query_shape[kIndex0] * head_num_value * query_shape[kIndex1] * key_shape[kIndex1];
      break;
    case mindspore::ops::FASInputLayoutMode::SBH:
      *numels = query_shape[kIndex1] * head_num_value * query_shape[kIndex0] * key_shape[kIndex0];
      break;
    case mindspore::ops::FASInputLayoutMode::BNSD:
      *numels = query_shape[kIndex0] * query_shape[kIndex1] * query_shape[kIndex2] * key_shape[kIndex2];
      break;
    case mindspore::ops::FASInputLayoutMode::BSND:
      *numels = query_shape[kIndex0] * query_shape[kIndex2] * query_shape[kIndex1] * key_shape[kIndex1];
      break;
    default:
      break;
  }

  std::optional<TensorPtr> dropout_mask = std::nullopt;
  if (keep_prob_value > 0.0 && keep_prob_value < 1.0) {
    auto p = std::make_shared<FP32Imm>(static_cast<float>(1 - keep_prob_value));
    auto shape = std::make_shared<ValueTuple>(std::vector<ValuePtr>{MakeValue<int64_t>(*numels)});
    auto dtype = std::make_shared<Int64Imm>(static_cast<int64_t>(query->Dtype()->type_id()));
    auto dropout_gen_mask_ext_op = CREATE_PYBOOST_OP(DropoutGenMaskExt, device::DeviceType::kAscend);
    dropout_mask.emplace(dropout_gen_mask_ext_op->Call(shape, p, seed, offset, dtype));
  } else if (keep_prob_value == 0) {
    constexpr int64_t kAlignBitNum = 128;
    constexpr int64_t kBitOfByte = 8;
    constexpr int64_t kAppendLen = 32;
    int64_t align_length = (*numels + kAlignBitNum - 1) / kAlignBitNum * kAlignBitNum / kBitOfByte;
    align_length += kAppendLen;
    auto shape = std::make_shared<ValueTuple>(std::vector<ValuePtr>{MakeValue(align_length)});
    auto zeros_op = CREATE_PYBOOST_OP(Zeros, device::DeviceType::kAscend);
    dropout_mask.emplace(zeros_op->Call(shape, std::make_shared<Int64Imm>(static_cast<int64_t>(kNumberTypeUInt8))));
  }

  return dropout_mask;
}

tensor::TensorPtr RecordRandomStateBeforeGenMask(const TensorPtr &tensor, double keep_prob_value) {
  // seed & offset will be inplace update after dropout_gen_mask
  if (keep_prob_value > 0.0 && keep_prob_value < 1.0) {
    auto tensor_cpu = tensor->cpu();
    int64_t value = *static_cast<int64_t *>(tensor_cpu->data_c());
    return tensor::from_scalar(value);
  }
  return tensor;
}
}  // namespace

void SpeedFusionAttentionAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &query, const TensorPtr &key, const TensorPtr &value,
  const Int64ImmPtr &head_num, const Int64ImmPtr &input_layout, const TensorPtr &seed, const TensorPtr &offset,
  const std::optional<TensorPtr> &pse, const std::optional<TensorPtr> &padding_mask,
  const std::optional<TensorPtr> &atten_mask, const FP32ImmPtr &scale, const FP32ImmPtr &keep_prob,
  const Int64ImmPtr &pre_tokens, const Int64ImmPtr &next_tokens, const Int64ImmPtr &inner_precise,
  const std::optional<ValueTuplePtr> &prefix, const std::optional<ValueTuplePtr> &actual_seq_qlen,
  const std::optional<ValueTuplePtr> &actual_seq_kvlen, const Int64ImmPtr &sparse_mode,
  const BoolImmPtr &gen_mask_parallel, const BoolImmPtr &sync, const Int64ImmPtr &pse_type,
  const std::optional<ValueTuplePtr> &q_start_idx, const std::optional<ValueTuplePtr> &kv_start_idx) {
  OpRunner::InferOpOutput(op, query, key, value, head_num, input_layout, seed, offset, pse, padding_mask, atten_mask,
                          scale, keep_prob, pre_tokens, next_tokens, inner_precise, prefix, actual_seq_qlen,
                          actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync, pse_type, q_start_idx, kv_start_idx);

  auto input_layout_value = static_cast<mindspore::ops::FASInputLayoutMode>(GetValue<int64_t>(input_layout));
  auto actual_seq_qlen_array = ConvertValueTupleToVector<int64_t>(actual_seq_qlen);
  auto actual_seq_kvlen_array = ConvertValueTupleToVector<int64_t>(actual_seq_kvlen);
  int64_t numels = 0;

  if (input_layout_value == mindspore::ops::FASInputLayoutMode::TND &&
      actual_seq_qlen_array.size() == actual_seq_kvlen_array.size()) {
    numels = query->shape()[kIndex1];
    int64_t accumulation = actual_seq_qlen_array[kIndex0] * actual_seq_kvlen_array[kIndex0];
    for (size_t idx = kIndex1; idx < actual_seq_qlen_array.size(); ++idx) {
      accumulation += ((actual_seq_qlen_array[idx] - actual_seq_qlen_array[idx - 1]) *
                       (actual_seq_kvlen_array[idx] - actual_seq_kvlen_array[idx - 1]));
    }
    numels *= accumulation;
  }

  auto head_num_value = GetValue<int64_t>(head_num);
  auto keep_prob_value = static_cast<double>(GetValue<float>(keep_prob));
  auto ori_seed = RecordRandomStateBeforeGenMask(seed, keep_prob_value);
  auto ori_offset = RecordRandomStateBeforeGenMask(offset, keep_prob_value);
  auto dropout_mask =
    SpeedFusionAttentionDropoutGenMaskCall(op, query, key, head_num_value, input_layout_value, keep_prob_value, seed,
                                           offset, gen_mask_parallel, sync, &numels);

  auto input_layout_str =
    mindspore::device::ascend::FASInputLayoutMode::ConvertEnumToString(GetValue<int64_t>(input_layout));
  auto scale_value = static_cast<double>(GetValue<float>(scale));
  auto pre_tokens_value = GetValue<int64_t>(pre_tokens);
  auto next_tokens_value = GetValue<int64_t>(next_tokens);
  auto inner_precise_value = GetValue<int64_t>(inner_precise);
  auto sparse_mode_value = GetValue<int64_t>(sparse_mode);
  auto pse_type_value = GetValue<int64_t>(pse_type);
  auto prefix_array = ConvertValueTupleToVector<int64_t>(prefix);
  auto q_start_idx_array = ConvertValueTupleToVector<int64_t>(q_start_idx);
  auto kv_start_idx_array = ConvertValueTupleToVector<int64_t>(kv_start_idx);

  constexpr int64_t kHostOutputNum = 3;
  auto device_outputs = std::vector<tensor::TensorPtr>(op->outputs().begin(), op->outputs().end() - kHostOutputNum);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), query, key, value, pse, dropout_mask,
                                padding_mask, atten_mask);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), device_outputs);

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, query, key, value, head_num_value, input_layout_str, pse, dropout_mask, padding_mask, atten_mask, scale_value,
     keep_prob_value, pre_tokens_value, next_tokens_value, inner_precise_value, prefix_array, actual_seq_qlen_array,
     actual_seq_kvlen_array, sparse_mode_value, pse_type_value, q_start_idx_array, kv_start_idx_array,
     device_outputs]() {
      auto device_context = op->device_context();
      const auto &outputs = device_outputs;
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, query, key, value, pse, dropout_mask, padding_mask, atten_mask);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
      if (!actual_seq_qlen_array.empty() && !actual_seq_kvlen_array.empty()) {
        LAUNCH_ACLNN(aclnnFlashAttentionVarLenScoreV2, device_context, op->stream_id(), query, key, value, pse,
                     dropout_mask, padding_mask, atten_mask, prefix_array, actual_seq_qlen_array,
                     actual_seq_kvlen_array, q_start_idx_array, kv_start_idx_array, scale_value, keep_prob_value,
                     pre_tokens_value, next_tokens_value, head_num_value, input_layout_str, inner_precise_value,
                     sparse_mode_value, pse_type_value, outputs[kIndex1], outputs[kIndex2], outputs[kIndex3],
                     outputs[kIndex0]);
      } else {
        LAUNCH_ACLNN(aclnnFlashAttentionScoreV2, device_context, op->stream_id(), query, key, value, pse, dropout_mask,
                     padding_mask, atten_mask, prefix_array, q_start_idx_array, kv_start_idx_array, scale_value,
                     keep_prob_value, pre_tokens_value, next_tokens_value, head_num_value, input_layout_str,
                     inner_precise_value, sparse_mode_value, pse_type_value, outputs[kIndex1], outputs[kIndex2],
                     outputs[kIndex3], outputs[kIndex0]);
      }
      MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
    }));

  // Set host outputs
  device_outputs.emplace_back(ori_seed);
  device_outputs.emplace_back(ori_offset);
  device_outputs.emplace_back(tensor::from_scalar(numels));
  op->set_outputs(device_outputs);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
