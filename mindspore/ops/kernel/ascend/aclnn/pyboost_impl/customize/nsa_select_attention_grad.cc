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

#include "kernel/ascend/aclnn/pyboost_impl/customize/nsa_select_attention_grad.h"
#include <string>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr NsaSelectAttentionGradAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &grad, const TensorPtr &query, const TensorPtr &key,
  const TensorPtr &value, const TensorPtr &attention_out, const TensorPtr &softmax_max, const TensorPtr &softmax_sum,
  const TensorPtr &topk_indices, const FP32ImmPtr &scale_value, const Int64ImmPtr &head_num,
  const Int64ImmPtr &select_block_size, const Int64ImmPtr &select_block_count,
  const std::optional<TensorPtr> &atten_mask, const std::optional<ValueTuplePtr> &actual_seq_qlen,
  const std::optional<ValueTuplePtr> &actual_seq_kvlen) {
  MS_LOG(DEBUG) << "NsaSelectAttentionGrad call start";
  OpRunner::InferOpOutput(op, grad, query, key, value, attention_out, softmax_max, softmax_sum, topk_indices,
                          scale_value, head_num, select_block_size, select_block_count, atten_mask, actual_seq_qlen,
                          actual_seq_kvlen);

  auto scale_value_imm = static_cast<double>(scale_value->value());
  auto head_num_imm = head_num->value();
  auto select_block_size_imm = select_block_size->value();
  auto select_block_count_imm = select_block_count->value();
  auto actual_seq_qlen_array = ConvertValueTupleToVector<int64_t>(actual_seq_qlen);
  auto actual_seq_kvlen_array = ConvertValueTupleToVector<int64_t>(actual_seq_kvlen);
  auto sparse_mode = 2LL;
  std::string layout_str = "TND";

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), grad, query, key, value, attention_out,
                                softmax_max, softmax_sum, topk_indices, atten_mask);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, grad, query, key, value, attention_out, softmax_max, softmax_sum, topk_indices, scale_value_imm, head_num_imm,
     select_block_size_imm, select_block_count_imm, atten_mask, actual_seq_qlen_array, actual_seq_kvlen_array,
     sparse_mode, layout_str]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();

      PyBoostUtils::MallocOpInputs(device_context, grad, query, key, value, attention_out, softmax_max, softmax_sum,
                                   topk_indices, atten_mask);
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
      LAUNCH_ACLNN(aclnnNsaSelectedAttentionGrad, device_context, op->stream_id(), query, key, value, attention_out,
                   grad, softmax_max, softmax_sum, topk_indices, actual_seq_qlen_array, actual_seq_kvlen_array,
                   atten_mask, scale_value_imm, select_block_size_imm, select_block_count_imm, head_num_imm, layout_str,
                   sparse_mode, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
      MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
