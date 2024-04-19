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

#include "plugin/device/ascend/kernel/pyboost/customize/one_hot_ext.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "kernel/pyboost/auto_generate/max.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr OneHotExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &tensor_tensor,
                                               const Int64ImmPtr &num_classes, const BaseTensorPtr &on_value,
                                               const BaseTensorPtr &off_value, const Int64ImmPtr &axis) {
  static const int64_t MIN_DEPTH = 1;
  static const int64_t AUTO_DEPTH = -1;

  int64_t num_class_imm = GetValue<int64_t>(num_classes);
  if (num_class_imm == AUTO_DEPTH) {
    auto max_op = CREATE_PYBOOST_OP(Max, op->device_context()->device_context_key_.device_name_);
    auto max_tensor = max_op->Call(tensor_tensor);
    max_tensor->data_sync();
    auto max_data = static_cast<int64_t *>(max_tensor->data_c());
    num_class_imm = *max_data + 1;
    if (num_class_imm < MIN_DEPTH) {
      num_class_imm = MIN_DEPTH;
    }
  }
  auto num_class_new = std::make_shared<Int64Imm>(static_cast<int64_t>(num_class_imm));
  OpRunner::InferOpOutput(op, tensor_tensor, num_class_new, on_value, off_value, axis);
  // No need to convert input
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), tensor_tensor, on_value, off_value);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, tensor_tensor, num_class_imm, on_value, off_value]() {
      MS_LOG(DEBUG) << "Run device task OneHotExt start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      int64_t axis_imm = -1;
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, tensor_tensor, on_value, off_value);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnOneHot, device_context, op->stream_id(), tensor_tensor, num_class_imm, on_value, off_value,
                   axis_imm, outputs[0]);
      MS_LOG(DEBUG) << "Run device task OneHotExt end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
