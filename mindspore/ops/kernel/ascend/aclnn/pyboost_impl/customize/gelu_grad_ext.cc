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
 * limitations under the License.∏
 */

#include "kernel/ascend/aclnn/pyboost_impl/customize/gelu_grad_ext.h"

#include <string>
#include <unordered_map>

#include "ir/scalar.h"
#include "include/op_enum.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
using ops::Approximate;
const std::unordered_map<Approximate, std::string> ApproximateModeMap{{Approximate::NONE, "none"},
                                                                      {Approximate::TANH, "tanh"}};
std::string GetApproximateMode(int64_t approximate) {
  auto approximate_enum = static_cast<Approximate>(approximate);
  auto it = ApproximateModeMap.find(approximate_enum);
  if (it == ApproximateModeMap.end()) {
    MS_EXCEPTION(ValueError) << "The value of approximate should be 0 or 1, but got " << approximate;
  }
  return it->second;
}
}  // namespace
tensor::TensorPtr GeluGradExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &grad_tensor,
                                             const TensorPtr &input_tensor, const Int64ImmPtr &approximate) {
  OpRunner::InferOpOutput(op, grad_tensor, input_tensor, approximate);

  auto approximate_value = GetValue<int64_t>(approximate);
  auto mode = GetApproximateMode(approximate_value);

  // Create device address for input/output tensors
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), grad_tensor, input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, grad_tensor, input_tensor, mode]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, grad_tensor, input_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
    LAUNCH_ACLNN(aclnnGeluBackwardV2, device_context, op->stream_id(), grad_tensor, input_tensor, mode, outputs[0]);
    MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
