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

#include "plugin/device/ascend/kernel/pyboost/customize/group_norm.h"
#include <memory>
#include <functional>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "kernel/pyboost/auto_generate/contiguous.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
constexpr size_t kNumberTwo = 2;
}  // namespace
void GroupNormAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                              const Int64ImmPtr &num_groups, const std::optional<TensorPtr> &gamma_opt_tensor,
                              const std::optional<TensorPtr> &beta_opt_tensor, const FP32ImmPtr &eps) {
  MS_LOG(DEBUG) << "Call start";
  // Convert ValuePtr to c++ scalar
  OpRunner::InferOpOutput(op, input_tensor, num_groups, gamma_opt_tensor, beta_opt_tensor, eps);
  auto num_groups_imm = GetValue<int64_t>(num_groups);
  auto eps_imm = static_cast<double>(GetValue<float>(eps));

  const auto &shape = input_tensor->shape();
  const int64_t N = shape[0];
  const int64_t C = shape[1];
  const int64_t HxW =
    (shape.size() == kNumberTwo) ? 1 : std::accumulate(shape.begin() + 2, shape.end(), 1, std::multiplies<int64_t>());
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, gamma_opt_tensor, beta_opt_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, input_tensor, gamma_opt_tensor, beta_opt_tensor, N, C, HxW, num_groups_imm, eps_imm]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor, gamma_opt_tensor, beta_opt_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnGroupNorm, device_context, op->stream_id(), input_tensor, gamma_opt_tensor, beta_opt_tensor, N,
                   C, HxW, num_groups_imm, eps_imm, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
      MS_LOG(DEBUG) << "Launch end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
