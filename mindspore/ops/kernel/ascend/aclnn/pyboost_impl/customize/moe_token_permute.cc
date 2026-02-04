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

#include "kernel/ascend/aclnn/pyboost_impl/customize/moe_token_permute.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <tuple>
#include "include/utils/utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr> MoeTokenPermuteAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &tokens, const TensorPtr &indices,
  const std::optional<Int64ImmPtr> &num_out_tokens, const BoolImmPtr &padded_mode) {
  OpRunner::InferOpOutput(op, tokens, indices, num_out_tokens, padded_mode);
  // Convert ValuePtr to c++ scalar
  auto padded_mode_imm = GetValue<bool>(padded_mode);
  auto num_out_tokens_imm = num_out_tokens.has_value() ? GetValue<int64_t>(num_out_tokens.value()) : 0;
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), tokens, indices);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, tokens, indices, num_out_tokens_imm, padded_mode_imm]() {
      MS_LOG(DEBUG) << "Run device task MoeTokenPermute start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(op->device_context(), tokens, indices);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(op->device_context(), outputs);
      LAUNCH_ACLNN(aclnnMoeTokenPermute, device_context, op->stream_id(), tokens, indices, num_out_tokens_imm,
                   padded_mode_imm, outputs[0], outputs[1]);
      MS_LOG(DEBUG) << "Run device task MoeTokenPermute end";
    }));
  return std::make_tuple(op->output(0), op->output(1));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
