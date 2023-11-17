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

#include "plugin/device/ascend/kernel/pyboost/customize/upsample_nearest1d.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/py_boost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
tensor::TensorPtr UpsampleNearest1dAscendCall(const PrimitivePtr &primitive,
                                              const device::DeviceContext *device_context,
                                              const TensorPtr &input_tensor, const std::vector<int64_t> &output_size,
                                              const std::vector<tensor::TensorPtr> &outputs) {
  MS_LOG(DEBUG) << "Call start";
  auto stream_ptr = device_context->device_res_manager_->GetStream(kDefaultStreamIndex);
  LAUNCH_ACLNN(aclnnUpsampleNearest1d, device_context, stream_ptr, input_tensor, output_size, outputs[0]);
  return outputs[0];
}
}  // namespace

tensor::TensorPtr UpsampleNearest1dAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                                   const ValueTuplePtr &output_size,
                                                   const ValueTuplePtr &scale_factors) {
  OpRunner::InferOpOutput(op, input_tensor, output_size, scale_factors);

  std::vector<int64_t> output_size_vector = ConvertValueTupleToVector<int64_t>(output_size);

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<pynative::PyBoostDeviceTask>([op, input_tensor, output_size_vector]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::PrepareOpInputs(device_context, input_tensor);
    // Malloc for output tensors
    PyBoostUtils::PrepareOpOutputs(device_context, outputs, op->device_sync_promises());
    UpsampleNearest1dAscendCall(op->primitive(), device_context, input_tensor, output_size_vector, outputs);
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
