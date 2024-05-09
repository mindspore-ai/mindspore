/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/pyboost/customize/grid_sampler_3d_grad.h"
#include <tuple>
#include <string>
#include <memory>
#include <vector>
#include "ops/auto_generate/gen_ops_primitive.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::BaseTensorPtr, tensor::BaseTensorPtr> GridSampler3DGradAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &grad_tensor, const BaseTensorPtr &input_x_tensor,
  const BaseTensorPtr &grid_tensor, const Int64ImmPtr &interpolation_mode, const Int64ImmPtr &padding_mode,
  const BoolImmPtr &align_corners) {
  constexpr char op_name[] = "GridSampler3DGrad";
  MS_LOG(DEBUG) << op_name << " call start";
  auto device_context = op->device_context();
  OpRunner::InferOpOutput(op, grad_tensor, input_x_tensor, grid_tensor, interpolation_mode, padding_mode,
                          align_corners);
  // ValueTuple to std::vector

  // Convert ValuePtr to c++ scalar
  // Convert ValuePtr to c++ scalar
  auto interpolation_mode_imm = GetValue<int64_t>(interpolation_mode);
  auto padding_mode_imm = GetValue<int64_t>(padding_mode);
  auto align_corners_imm = GetValue<bool>(align_corners);

  PyBoostUtils::PrepareOpInputs(device_context, op->stream_id(), grad_tensor, input_x_tensor, grid_tensor);

  PyBoostUtils::PrepareOpOutputs(device_context, op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, grad_tensor, input_x_tensor, grid_tensor, interpolation_mode_imm,
                                                  padding_mode_imm, align_corners_imm, op_name]() {
      MS_LOG(DEBUG) << "Run device task " << op_name << " end";
      auto device_context = op->device_context();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, grad_tensor, input_x_tensor, grid_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, op->outputs());

      std::vector<uint8_t> output_mask{1, 1};
      LAUNCH_ACLNN(aclnnGridSampler3DBackward, device_context, op->stream_id(), grad_tensor, input_x_tensor,
                   grid_tensor, interpolation_mode_imm, padding_mode_imm, align_corners_imm, output_mask, op->output(0),
                   op->output(1));
      MS_LOG(DEBUG) << "Run device task " << op_name << " end";
    }));
  return std::make_tuple(op->output(0), op->output(1));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
