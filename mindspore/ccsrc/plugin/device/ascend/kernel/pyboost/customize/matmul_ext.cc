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

#include "plugin/device/ascend/kernel/pyboost/customize/matmul_ext.h"
#include <string>
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void MatMulExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                              const BaseTensorPtr &mat2_tensor) {
  OpRunner::InferOpOutput(op, input_tensor, mat2_tensor);
  // No need to convert input
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, mat2_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, mat2_tensor]() {
    MS_LOG(DEBUG) << "Run device task BatchMatMulExt start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor, mat2_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    // cubeMathType: 0 - KEEP_DTYPE, 1 - ALLOW_FP32_DOWN_PRECISION
    auto cube_math_type = GetCubeMathType();
    LAUNCH_ACLNN(aclnnMatmul, device_context, op->stream_id(), input_tensor, mat2_tensor, outputs[0], cube_math_type);
    MS_LOG(DEBUG) << "Run device task BatchMatMulExt end";
  }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
