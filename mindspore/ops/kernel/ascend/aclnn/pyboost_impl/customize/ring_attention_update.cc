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

#include "kernel/ascend/aclnn/pyboost_impl/customize/ring_attention_update.h"
#include <string>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
#include "include/op_enum.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> RingAttentionUpdateAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &prev_attn_out, const TensorPtr &prev_softmax_max,
  const TensorPtr &prev_softmax_sum, const TensorPtr &cur_attn_out, const TensorPtr &cur_softmax_max,
  const TensorPtr &cur_softmax_sum, const std::optional<TensorPtr> &actual_seq_qlen, const Int64ImmPtr layout) {
  OpRunner::InferOpOutput(op, prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out, cur_softmax_max,
                          cur_softmax_sum, actual_seq_qlen, layout);
  auto layout_str = mindspore::device::ascend::FASInputLayoutMode::ConvertEnumToString(GetValue<int64_t>(layout));
  if (layout_str != "SBH" && layout_str != "TND") {
    MS_EXCEPTION(ValueError) << "For RingAttentionUpdate, the value of 'layout' must be SBH/TND.";
  }
  if (layout_str == "TND" && !actual_seq_qlen.has_value()) {
    MS_EXCEPTION(ValueError) << "For RingAttentionUpdate, 'actual_seq_qlen' must has value when layout is 'TND'.";
  }

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), prev_attn_out, prev_softmax_max,
                                prev_softmax_sum, cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out,
                                                  cur_softmax_max, cur_softmax_sum, actual_seq_qlen, layout_str]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out,
                                   cur_softmax_max, cur_softmax_sum, actual_seq_qlen);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
      LAUNCH_ACLNN(aclnnRingAttentionUpdate, device_context, op->stream_id(), prev_attn_out, prev_softmax_max,
                   prev_softmax_sum, cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen, layout_str,
                   outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
      MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
    }));
  return std::make_tuple(op->output(kIndex0), op->output(kIndex1), op->output(kIndex2));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
