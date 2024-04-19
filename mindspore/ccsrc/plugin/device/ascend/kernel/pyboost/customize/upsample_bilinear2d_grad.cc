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

#include "plugin/device/ascend/kernel/pyboost/customize/upsample_bilinear2d_grad.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "mindapi/base/types.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
tensor::BaseTensorPtr UpsampleBilinear2DGradAscendCall(
  const std::shared_ptr<OpRunner> &op, const device::DeviceContext *device_context, const BaseTensorPtr &grad_out,
  const std::vector<int64_t> &input_size, const std::vector<int64_t> &output_size, const std::vector<pyfloat> &scales,
  const bool &align_corners, const std::vector<tensor::BaseTensorPtr> &outputs) {
  MS_LOG(DEBUG) << "Call start";
  double scales_h = scales[0];
  double scales_w = scales[1];
  LAUNCH_ACLNN(aclnnUpsampleBilinear2dBackward, device_context, op->stream_id(), grad_out, output_size, input_size,
               align_corners, scales_h, scales_w, outputs[0]);
  MS_LOG(DEBUG) << "Call end";
  return outputs[0];
}
}  // namespace

tensor::BaseTensorPtr UpsampleBilinear2DGradAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                            const BaseTensorPtr &gradout_tensor,
                                                            const ValueTuplePtr &input_size,
                                                            const std::optional<ValueTuplePtr> &output_size,
                                                            const std::optional<ValueTuplePtr> &scale_factors,
                                                            const BoolImmPtr &align_corners) {
  MS_LOG(DEBUG) << "UpsampleBilinear2DGradAscendCustomize start";
  OpRunner::InferOpOutput(op, gradout_tensor, input_size, output_size, scale_factors, align_corners);

  auto input_size_vector = ConvertValueTupleToVector<int64_t>(input_size);

  std::vector<int64_t> output_size_vector{};
  std::vector<pyfloat> scales{0., 0.};
  if (output_size.has_value()) {
    output_size_vector = ConvertValueTupleToVector<int64_t>(output_size.value());
  } else if (scale_factors.has_value()) {
    MS_EXCEPTION(RuntimeError) << "For UpsampleBilinear2DGrad, scale_factors is not supported now.";
    scales = ConvertValueTupleToVector<pyfloat>(scale_factors.value());
    for (size_t i = 0; i < scales.size(); ++i) {
      output_size_vector.push_back(static_cast<int64_t>(input_size_vector[i + kDim2]) * scales[i]);
    }
  }

  auto align_corners_val = GetValue<bool>(align_corners);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), gradout_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, gradout_tensor, input_size_vector, output_size_vector, scales, align_corners_val]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, gradout_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      // Call aclnnUpsampleBilinear2dBackward
      UpsampleBilinear2DGradAscendCall(op, device_context, gradout_tensor, input_size_vector, output_size_vector,
                                       scales, align_corners_val, outputs);
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
