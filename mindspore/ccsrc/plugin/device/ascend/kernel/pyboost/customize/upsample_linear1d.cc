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

#include "plugin/device/ascend/kernel/pyboost/customize/upsample_linear1d.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "mindapi/base/types.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
const pyfloat DEFAULT_SCALE_VALUE = -1;
tensor::BaseTensorPtr UpsampleLinear1DAscendCall(const std::shared_ptr<OpRunner> &op,
                                                 const device::DeviceContext *device_context,
                                                 const BaseTensorPtr &input_tensor,
                                                 const std::vector<int64_t> &output_size,
                                                 const std::vector<pyfloat> &scales, const bool &align_corners,
                                                 const std::vector<tensor::BaseTensorPtr> &outputs) {
  MS_LOG(DEBUG) << "Call start";
  double scales_l = scales[0];
  LAUNCH_ACLNN(aclnnUpsampleLinear1d, device_context, op->stream_id(), input_tensor, output_size, align_corners,
               scales_l, outputs[0]);
  MS_LOG(DEBUG) << "Call end";
  return outputs[0];
}
}  // namespace

tensor::BaseTensorPtr UpsampleLinear1DAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                      const BaseTensorPtr &input_tensor,
                                                      const std::optional<ValueTuplePtr> &output_size,
                                                      const std::optional<ValueTuplePtr> &scale_factors,
                                                      const BoolImmPtr &align_corners) {
  MS_LOG(INFO) << "UpsampleLinear1DAscendCustomize start";

  OpRunner::InferOpOutput(op, input_tensor, output_size, scale_factors, align_corners);

  // get output_size, scale_factors and align_corners
  const ShapeVector &osize = op->output_abs()->GetShape()->GetShapeVector();
  std::vector<int64_t> output_size_vector = {osize.begin() + kDim2, osize.end()};

  std::vector<pyfloat> scales{DEFAULT_SCALE_VALUE};
  if (scale_factors.has_value()) {
    scales = ConvertValueTupleToVector<pyfloat>(scale_factors.value());
  }

  auto align_corners_val = GetValue<bool>(align_corners);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, output_size_vector, scales, align_corners_val]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      // Call aclnnUpsampleLinear1d
      UpsampleLinear1DAscendCall(op, device_context, input_tensor, output_size_vector, scales, align_corners_val,
                                 outputs);
    }));

  MS_LOG(INFO) << "UpsampleLinear1DAscendCustomize end";

  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
