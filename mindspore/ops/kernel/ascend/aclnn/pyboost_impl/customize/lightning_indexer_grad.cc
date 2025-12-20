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

#include "kernel/ascend/aclnn/pyboost_impl/customize/lightning_indexer_grad.h"
#include <string>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
#include "mindspore/ops/op_def/op_enum.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void LightningIndexerGradAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &query,
                                         const TensorPtr &key, const TensorPtr &dy, const TensorPtr &sparse_indices,
                                         const TensorPtr &weights,
                                         const std::optional<TensorPtr> &actual_seq_lengths_query,
                                         const std::optional<TensorPtr> &actual_seq_lengths_key,
                                         const Int64ImmPtr &layout_query, const Int64ImmPtr &sparse_mode,
                                         const Int64ImmPtr &pre_tokens, const Int64ImmPtr &next_tokens) {
  OpRunner::InferOpOutput(op, query, key, dy, sparse_indices, weights, actual_seq_lengths_query, actual_seq_lengths_key,
                          layout_query, sparse_mode, pre_tokens, next_tokens);

  auto layout_query_str = device::ascend::FASInputLayoutMode::ConvertEnumToString(layout_query->value());
  auto sparse_mode_v = sparse_mode->value();
  auto pre_tokens_v = pre_tokens->value();
  auto next_tokens_v = next_tokens->value();
  int64_t head_num = 0;
  bool deterministic = true;

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), query, key, dy, sparse_indices, weights,
                                actual_seq_lengths_query, actual_seq_lengths_key);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, query, key, dy, sparse_indices, weights, actual_seq_lengths_query, actual_seq_lengths_key, layout_query_str,
     sparse_mode_v, pre_tokens_v, next_tokens_v, head_num, deterministic]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, query, key, dy, sparse_indices, weights, actual_seq_lengths_query,
                                   actual_seq_lengths_key);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
      LAUNCH_ACLNN(aclnnLightningIndexerGrad, device_context, op->stream_id(), query, key, dy, sparse_indices, weights,
                   actual_seq_lengths_query, actual_seq_lengths_key, head_num, layout_query_str, sparse_mode_v,
                   pre_tokens_v, next_tokens_v, deterministic, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
      MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
