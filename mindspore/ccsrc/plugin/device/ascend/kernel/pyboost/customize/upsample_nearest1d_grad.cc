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

#include "plugin/device/ascend/kernel/pyboost/customize/upsample_nearest1d_grad.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "mindapi/base/types.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
tensor::BaseTensorPtr UpsampleNearest1DGradAscendCall(
  const std::shared_ptr<OpRunner> &op, const device::DeviceContext *device_context, const BaseTensorPtr &grad_out,
  const std::vector<int64_t> &input_size, const std::vector<int64_t> &output_size, const std::vector<pyfloat> &scales,
  const std::vector<tensor::BaseTensorPtr> &outputs) {
  MS_LOG(DEBUG) << "Call start";
  double scales_l = scales[0];
  LAUNCH_ACLNN(aclnnUpsampleNearest1dBackward, device_context, op->stream_id(), grad_out, output_size, input_size,
               scales_l, outputs[0]);
  MS_LOG(DEBUG) << "Call end";
  return outputs[0];
}
}  // namespace

tensor::BaseTensorPtr UpsampleNearest1DGradAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                           const BaseTensorPtr &gradout_tensor,
                                                           const ValueTuplePtr &input_size,
                                                           const std::optional<ValueTuplePtr> &output_size,
                                                           const std::optional<ValueTuplePtr> &scale_factors) {
  MS_LOG(DEBUG) << "UpsampleNearest1DGradAscendCustomize start";
  OpRunner::InferOpOutput(op, gradout_tensor, input_size, output_size, scale_factors);

  auto input_size_vector = ConvertValueTupleToVector<int64_t>(input_size);

  std::vector<int64_t> output_size_vector{};
  std::vector<pyfloat> scales{0.};
  if (output_size.has_value()) {
    output_size_vector = ConvertValueTupleToVector<int64_t>(output_size.value());
  } else if (scale_factors.has_value()) {
    scales = ConvertValueTupleToVector<pyfloat>(scale_factors.value());
    for (size_t i = 0; i < scales.size(); ++i) {
      output_size_vector.push_back(static_cast<int64_t>(input_size_vector[i + kDim2]) * scales[i]);
    }
  }

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), gradout_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, gradout_tensor, input_size_vector, output_size_vector, scales]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, gradout_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      // Call aclnnUpsampleNearest1DGrad
      UpsampleNearest1DGradAscendCall(op, device_context, gradout_tensor, input_size_vector, output_size_vector, scales,
                                      outputs);
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
