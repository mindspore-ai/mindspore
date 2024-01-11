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

#include "plugin/device/ascend/kernel/pyboost/customize/group_norm_grad.h"
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

void GroupNormGradAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &dout_tensor,
                                  const TensorPtr &input_tensor, const TensorPtr &mean_tensor,
                                  const TensorPtr &rstd_tensor, const TensorPtr &weight_opt_tensor,
                                  const Int64ImmPtr &num_groups, const BoolImmPtr &dx_is_require,
                                  const BoolImmPtr &dgamma_is_require, const BoolImmPtr &dbeta_is_require) {
  MS_LOG(DEBUG) << "Call start";
  // Convert ValuePtr to c++ scalar
  OpRunner::InferOpOutput(op, dout_tensor, input_tensor, mean_tensor, rstd_tensor, weight_opt_tensor, num_groups,
                          dx_is_require, dgamma_is_require, dbeta_is_require);
  const auto &x_shape = input_tensor->shape();
  const int64_t N = x_shape[0];
  const int64_t C = x_shape[1];
  const int64_t HxW = (x_shape.size() == kNumberTwo)
                        ? 1
                        : std::accumulate(x_shape.begin() + 2, x_shape.end(), 1, std::multiplies<int64_t>());
  auto num_groups_imm = GetValue<int64_t>(num_groups);
  auto dx_require = GetValue<bool>(dx_is_require);
  auto dgamma_require = GetValue<bool>(dgamma_is_require);
  auto dbeta_require = GetValue<bool>(dbeta_is_require);

  std::vector<uint8_t> output_mask{};
  output_mask.emplace_back(static_cast<uint8_t>(dx_require));
  output_mask.emplace_back(static_cast<uint8_t>(dgamma_require));
  output_mask.emplace_back(static_cast<uint8_t>(dbeta_require));

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), dout_tensor, input_tensor, mean_tensor,
                                rstd_tensor, weight_opt_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, dout_tensor, input_tensor, mean_tensor, rstd_tensor,
                                                  weight_opt_tensor, N, C, HxW, num_groups_imm, output_mask]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, dout_tensor, input_tensor, mean_tensor, rstd_tensor,
                                   weight_opt_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnGroupNormBackward, device_context, op->stream_id(), dout_tensor, input_tensor, mean_tensor,
                   rstd_tensor, weight_opt_tensor, N, C, HxW, num_groups_imm, output_mask, outputs[kIndex0],
                   outputs[kIndex1], outputs[kIndex2]);
      MS_LOG(DEBUG) << "Launch end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
