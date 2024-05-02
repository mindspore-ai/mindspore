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

#include "plugin/device/ascend/kernel/pyboost/customize/avg_pool2d.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "mindapi/base/types.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
tensor::BaseTensorPtr AvgPool2DAscendCall(const std::shared_ptr<OpRunner> &op,
                                          const device::DeviceContext *device_context, const BaseTensorPtr &input,
                                          const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &stride,
                                          const std::vector<int64_t> &padding, const bool &ceil_mode,
                                          const bool count_include_pad, const int64_t divisor_override,
                                          const std::vector<tensor::BaseTensorPtr> &outputs) {
  MS_LOG(DEBUG) << "Call start";
  const int8_t cube_math_type = GetCubeMathType();
  LAUNCH_ACLNN(aclnnAvgPool2d, device_context, op->stream_id(), input, kernel_size, stride, padding, ceil_mode,
               count_include_pad, divisor_override, cube_math_type, outputs[0]);
  MS_LOG(DEBUG) << "Call end";
  return outputs[0];
}
}  // namespace

tensor::BaseTensorPtr AvgPool2DAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input,
                                               const ValueTuplePtr &kernel_size, const ValueTuplePtr &stride,
                                               const ValueTuplePtr &padding, const BoolImmPtr &ceil_mode,
                                               const BoolImmPtr &count_include_pad,
                                               const std::optional<Int64ImmPtr> &divisor_override) {
  MS_LOG(INFO) << "AvgPool2DAscendCustomize start";
  OpRunner::InferOpOutput(op, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);

  auto kernel_size_val = ConvertValueTupleToVector<int64_t>(kernel_size);
  auto stride_val = ConvertValueTupleToVector<int64_t>(stride);
  auto padding_val = ConvertValueTupleToVector<int64_t>(padding);
  auto ceil_mode_val = GetValue<bool>(ceil_mode);
  auto count_include_pad_val = GetValue<bool>(count_include_pad);
  auto divisor_override_val = divisor_override.has_value() ? GetValue<int64_t>(divisor_override.value()) : 0;

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input, kernel_size_val, stride_val, padding_val, ceil_mode_val,
                                                  count_include_pad_val, divisor_override_val]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      // Call aclnnAvgPool2d
      AvgPool2DAscendCall(op, device_context, input, kernel_size_val, stride_val, padding_val, ceil_mode_val,
                          count_include_pad_val, divisor_override_val, outputs);
    }));

  MS_LOG(INFO) << "AvgPool2DAscendCustomize end";

  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
