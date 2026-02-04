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

#include "kernel/ascend/aclnn/pyboost_impl/customize/divmods.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr DivModsAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input,
                                         const ScalarPtr &other, const std::optional<Int64ImmPtr> &rounding_mode) {
  OpRunner::InferOpOutput(op, input, other, rounding_mode);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  auto other_real = ops::FetchRealScalar(other);
  auto mode = 0;
  if (rounding_mode.has_value()) {
    mode = static_cast<int>(GetValue<int64_t>(rounding_mode.value()));
  }

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input, other_real, mode]() {
    MS_LOG(DEBUG) << "Run device task DivMods start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    LAUNCH_ACLNN(aclnnDivMods, device_context, op->stream_id(), input, other_real, mode, outputs[0]);
    MS_LOG(DEBUG) << "Run device task DivMods end";
  }));

  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
