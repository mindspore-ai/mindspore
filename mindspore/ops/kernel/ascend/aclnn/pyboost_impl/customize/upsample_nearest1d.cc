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

#include "kernel/ascend/aclnn/pyboost_impl/customize/upsample_nearest1d.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
tensor::TensorPtr UpsampleNearest1dAscendCall(const std::shared_ptr<OpRunner> &op,
                                              const device::DeviceContext *device_context,
                                              const TensorPtr &input_tensor, const std::vector<int64_t> &output_size,
                                              const std::vector<tensor::TensorPtr> &outputs) {
  MS_LOG(DEBUG) << "Call start";
  LAUNCH_ACLNN(aclnnUpsampleNearest1d, device_context, op->stream_id(), input_tensor, output_size, outputs[0]);
  return outputs[0];
}
}  // namespace

tensor::TensorPtr UpsampleNearest1DAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                                   const std::optional<ValueTuplePtr> &output_size,
                                                   const std::optional<ValueTuplePtr> &scale_factors) {
  OpRunner::InferOpOutput(op, input_tensor, output_size, scale_factors);

  const ShapeVector &osize = op->output(kIndex0)->shape();
  std::vector<int64_t> output_size_vector = {osize.begin() + kDim2, osize.end()};

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, output_size_vector]() {
    MS_LOG(DEBUG) << "Run device task UpsampleNearest1d start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    // Call aclnnUpsampleNearest1d
    UpsampleNearest1dAscendCall(op, device_context, input_tensor, output_size_vector, outputs);
    MS_LOG(DEBUG) << "Run device task UpsampleNearest1d end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
