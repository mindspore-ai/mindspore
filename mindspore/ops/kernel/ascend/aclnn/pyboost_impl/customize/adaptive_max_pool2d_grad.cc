/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/adaptive_max_pool2d_grad.h"
#include <memory>
#include <tuple>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr AdaptiveMaxPool2DGradAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &y_grad,
                                                       const TensorPtr &x, const TensorPtr &argmax) {
  OpRunner::InferOpOutput(op, y_grad, x, argmax);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), y_grad, x, argmax);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, y_grad, x, argmax]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, y_grad, x, argmax);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
    LAUNCH_ACLNN(aclnnAdaptiveMaxPool2dBackward, device_context, op->stream_id(), y_grad, x, argmax, outputs[0]);
    MS_LOG(DEBUG) << op->primitive()->name() << " Call end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
