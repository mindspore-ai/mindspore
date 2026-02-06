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

#include "kernel/ascend/aclnn/pyboost_impl/customize/grid_sampler_2d_grad.h"
#include <tuple>
#include <string>
#include <algorithm>
#include <memory>
#include <vector>
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr> GridSampler2DGradAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &grad_tensor, const TensorPtr &input_x_tensor,
  const TensorPtr &grid_tensor, const Int64ImmPtr &interpolation_mode, const Int64ImmPtr &padding_mode,
  const BoolImmPtr &align_corners, const ValueTuplePtr &output_mask) {
  constexpr char op_name[] = "GridSampler2DGrad";
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
  const auto &output_mask_tmp = ConvertValueTupleToVector<int64_t>(output_mask);
  std::vector<uint8_t> output_mask_vec;
  std::transform(output_mask_tmp.begin(), output_mask_tmp.end(), std::back_inserter(output_mask_vec),
                 [](int64_t value) { return static_cast<uint8_t>(value); });

  PyBoostUtils::PrepareOpInputs(device_context, op->stream_id(), grad_tensor, input_x_tensor, grid_tensor);

  PyBoostUtils::PrepareOpOutputs(device_context, op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, grad_tensor, input_x_tensor, grid_tensor, interpolation_mode_imm,
                                                  padding_mode_imm, align_corners_imm, output_mask_vec, op_name]() {
      MS_LOG(DEBUG) << "Run device task " << op_name << " end";
      auto device_context = op->device_context();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, grad_tensor, input_x_tensor, grid_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, op->outputs());

      LAUNCH_ACLNN(aclnnGridSampler2DBackward, device_context, op->stream_id(), grad_tensor, input_x_tensor,
                   grid_tensor, interpolation_mode_imm, padding_mode_imm, align_corners_imm, output_mask_vec,
                   op->output(0), op->output(1));
      MS_LOG(DEBUG) << "Run device task " << op_name << " end";
    }));
  return std::make_tuple(op->output(0), op->output(1));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
