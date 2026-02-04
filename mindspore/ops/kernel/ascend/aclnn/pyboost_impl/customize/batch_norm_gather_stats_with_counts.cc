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

#include "kernel/ascend/aclnn/pyboost_impl/customize/batch_norm_gather_stats_with_counts.h"
#include <memory>
#include <tuple>
#include <string>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr> BatchNormGatherStatsWithCountsAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor, const TensorPtr &mean_tensor,
  const TensorPtr &invstd_tensor, const std::optional<TensorPtr> &running_mean_tensor,
  const std::optional<TensorPtr> &running_var_tensor, const FP32ImmPtr &momentum, const FP32ImmPtr &eps,
  const std::optional<TensorPtr> &counts_tensor) {
  std::string op_name = op->primitive()->name();
  MS_LOG(DEBUG) << op_name << " call start";
  OpRunner::InferOpOutput(op, input_tensor, mean_tensor, invstd_tensor, running_mean_tensor, running_var_tensor,
                          momentum, eps, counts_tensor);
  // ValueTuple to std::vector

  // Convert ValuePtr to c++ scalar
  // Convert ValuePtr to c++ scalar
  auto momentum_imm = GetValue<float>(momentum);
  auto eps_imm = GetValue<float>(eps);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, mean_tensor, invstd_tensor,
                                running_mean_tensor, running_var_tensor, counts_tensor);

  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, mean_tensor, invstd_tensor, running_mean_tensor,
                                                  running_var_tensor, momentum_imm, eps_imm, counts_tensor]() {
      std::string op_name = op->primitive()->name();
      MS_LOG(DEBUG) << "Run device task " << op_name << " end";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor, mean_tensor, invstd_tensor, running_mean_tensor,
                                   running_var_tensor, counts_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      double momentum_imm_d = static_cast<double>(momentum_imm);
      double eps_imm_d = static_cast<double>(eps_imm);
      LAUNCH_ACLNN(aclnnBatchNormGatherStatsWithCounts, device_context, op->stream_id(), input_tensor, mean_tensor,
                   invstd_tensor, running_mean_tensor, running_var_tensor, momentum_imm_d, eps_imm_d, counts_tensor,
                   outputs[0], outputs[1]);
      MS_LOG(DEBUG) << "Run device task " << op_name << " end";
    }));
  return std::make_tuple(op->output(0), op->output(1));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
