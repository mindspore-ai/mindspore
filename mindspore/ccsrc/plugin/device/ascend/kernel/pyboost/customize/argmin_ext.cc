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

#include "plugin/device/ascend/kernel/pyboost/customize/argmin_ext.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "kernel/pyboost/auto_generate/reshape.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr ArgMinAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                            const std::optional<Int64ImmPtr> &dim, const BoolImmPtr &keepdim) {
  MS_LOG(DEBUG) << "aclnnArgmin call start";
  OpRunner::InferOpOutput(op, input_tensor, dim, keepdim);

  int64_t real_dim = 0;
  bool dim_is_none = true;
  auto real_keepdim = false;
  BaseTensorPtr real_input;
  if (dim.has_value()) {
    dim_is_none = false;
    real_dim = GetValue<int64_t>(dim.value());
    real_keepdim = GetValue<bool>(keepdim);
    real_input = input_tensor;
  } else {
    auto reshape_op = CREATE_PYBOOST_OP(Reshape, op->device_context()->device_context_key_.device_name_);
    real_input = reshape_op->Call(
      input_tensor, std::make_shared<ValueTuple>(std::vector<ValuePtr>({std::make_shared<Int64Imm>(-1)})));
  }

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), real_input);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, real_input, real_dim, real_keepdim, dim_is_none]() {
      MS_LOG(DEBUG) << "Run device task ArgMin start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, real_input);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      LAUNCH_ACLNN(aclnnArgMin, device_context, op->stream_id(), real_input, real_dim, real_keepdim, outputs[0]);
      MS_LOG(DEBUG) << "Run device task ArgMin end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
