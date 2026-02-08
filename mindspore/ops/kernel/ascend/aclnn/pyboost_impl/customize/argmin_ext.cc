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

#include "kernel/ascend/aclnn/pyboost_impl/customize/argmin_ext.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr ArgMinAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                        const std::optional<Int64ImmPtr> &dim, const BoolImmPtr &keepdim) {
  MS_LOG(DEBUG) << "aclnnArgmin call start";
  OpRunner::InferOpOutput(op, input_tensor, dim, keepdim);

  int64_t real_dim = 0;
  bool dim_is_none = true;
  auto real_keepdim = false;
  TensorPtr real_input;
  if (dim.has_value()) {
    dim_is_none = false;
    real_dim = GetValue<int64_t>(dim.value());
    real_keepdim = GetValue<bool>(keepdim);
    real_input = input_tensor;
  } else {
    kernel::pyboost::RequireGradGuard require_grad_guard(false);
    real_input = reshape(input_tensor, {-1});
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
