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

#include "plugin/device/ascend/kernel/pyboost/customize/max_pool_with_mask.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "runtime/device/device_address_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void MaxPoolWithMaskAscendCall(const std::shared_ptr<OpRunner> &op, const device::DeviceContext *device_context,
                               const BaseTensorPtr &x_tensor, const ValueTuplePtr &kernel_size,
                               const std::optional<ValueTuplePtr> &strides, const ValueTuplePtr &pads,
                               const ValueTuplePtr &dilation, const BoolImmPtr &ceil_mode,
                               const std::vector<tensor::BaseTensorPtr> &outputs) {
  std::vector<int64_t> strides_array;
  if (strides.has_value()) {
    strides_array = ConvertValueTupleToVector<int64_t>(strides.value());
  }
  auto kernel_size_array = ConvertValueTupleToVector<int64_t>(kernel_size);
  auto pads_array = ConvertValueTupleToVector<int64_t>(pads);
  auto dilation_array = ConvertValueTupleToVector<int64_t>(dilation);
  auto ceil_mode_scalar = GetValue<bool>(ceil_mode);
  LAUNCH_ACLNN(aclnnMaxPool2dWithMask, device_context, op->stream_id(), x_tensor, kernel_size_array, strides_array,
               pads_array, dilation_array, ceil_mode_scalar, outputs[0], outputs[1]);
}
}  // namespace

tensor::BaseTensorPtr MaxPoolWithMaskAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &x_tensor,
                                                     const ValueTuplePtr &kernel_size,
                                                     const std::optional<ValueTuplePtr> &strides,
                                                     const ValueTuplePtr &pads, const ValueTuplePtr &dilation,
                                                     const BoolImmPtr &ceil_mode, const Int64ImmPtr &argmax_type) {
  OpRunner::InferOpOutput(op, x_tensor, kernel_size, strides, pads, dilation, ceil_mode, argmax_type);
  // Create device address for input/output tensors
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, x_tensor, kernel_size, strides, pads, dilation, ceil_mode]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, x_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
      MaxPoolWithMaskAscendCall(op, device_context, x_tensor, kernel_size, strides, pads, dilation, ceil_mode, outputs);
      MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
