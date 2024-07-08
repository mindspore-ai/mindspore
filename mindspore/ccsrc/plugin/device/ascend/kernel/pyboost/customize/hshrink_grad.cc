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

#include "plugin/device/ascend/kernel/pyboost/customize/hshrink_grad.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "transform/acl_ir/acl_helper.h"
#include "plugin/device/ascend/kernel/acl/acl_kernel_mod.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr HShrinkGradAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                 const BaseTensorPtr &gradients_tensor,
                                                 const BaseTensorPtr &features_tensor, const ScalarPtr &lambd) {
  MS_LOG(DEBUG) << "HShrinkGrad Ascend start";
  OpRunner::InferOpOutput(op, gradients_tensor, features_tensor, lambd);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), gradients_tensor, features_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, gradients_tensor, features_tensor, lambd]() {
      MS_LOG(DEBUG) << "Run device task HShrinkGrad start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, gradients_tensor, features_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      LAUNCH_ACLNN(aclnnHardshrinkBackward, device_context, op->stream_id(), gradients_tensor, features_tensor, lambd,
                   outputs[0]);
      MS_LOG(DEBUG) << "Run device task HShrinkGrad end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
