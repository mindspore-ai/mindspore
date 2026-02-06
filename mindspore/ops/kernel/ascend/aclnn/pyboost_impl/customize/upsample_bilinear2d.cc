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

#include "kernel/ascend/aclnn/pyboost_impl/customize/upsample_bilinear2d.h"
#include "include/utils/utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "mindapi/base/types.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
tensor::TensorPtr UpsampleBilinear2DAscendCall(const std::shared_ptr<OpRunner> &op,
                                               const device::DeviceContext *device_context,
                                               const TensorPtr &input_tensor, const std::vector<int64_t> &output_size,
                                               const std::vector<pyfloat> &scales, const bool &align_corners,
                                               const std::vector<tensor::TensorPtr> &outputs) {
  MS_LOG(DEBUG) << "Call start";
  double scales_h = scales[0];
  double scales_w = scales[1];
  LAUNCH_ACLNN(aclnnUpsampleBilinear2d, device_context, op->stream_id(), input_tensor, output_size, align_corners,
               scales_h, scales_w, outputs[0]);
  MS_LOG(DEBUG) << "Call end";
  return outputs[0];
}
}  // namespace

tensor::TensorPtr UpsampleBilinear2DAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                                    const std::optional<ValueTuplePtr> &output_size,
                                                    const std::optional<ValueTuplePtr> &scale_factors,
                                                    const BoolImmPtr &align_corners) {
  MS_LOG(INFO) << "UpsampleBilinear2DAscendCustomize start";

  OpRunner::InferOpOutput(op, input_tensor, output_size, scale_factors, align_corners);

  // get output_size, scale_factors and align_corners
  const ShapeVector &osize = op->output(kIndex0)->shape();
  std::vector<int64_t> output_size_vector = {osize.begin() + kDim2, osize.end()};

  constexpr pyfloat DEFAULT_SCALE_VALUE = 0.;
  std::vector<pyfloat> scales(kDim2, DEFAULT_SCALE_VALUE);
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
      // Call aclnnUpsampleBilinear2d
      UpsampleBilinear2DAscendCall(op, device_context, input_tensor, output_size_vector, scales, align_corners_val,
                                   outputs);
    }));

  MS_LOG(INFO) << "UpsampleBilinear2DAscendCustomize end";

  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
