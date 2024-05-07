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

#include "plugin/device/ascend/kernel/pyboost/customize/searchsorted.h"

#include <memory>
#include <tuple>
#include <string>

#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr SearchSortedAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                  const BaseTensorPtr &sorted_sequence, const BaseTensorPtr &values,
                                                  const std::optional<BaseTensorPtr> &sorter, const Int64ImmPtr &dtype,
                                                  const BoolImmPtr &right) {
  OpRunner::InferOpOutput(op, sorted_sequence, values, sorter, dtype, right);
  // Convert ValuePtr to c++ scalar
  auto in_dtype = GetValue<int64_t>(dtype);
  auto in_right = GetValue<bool>(right);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), sorted_sequence, values, sorter);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, sorted_sequence, values, sorter, in_dtype, in_right]() {
      MS_LOG(DEBUG) << "Run device task SearchSorted start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, sorted_sequence, values, sorter);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      auto in_dtype_id = static_cast<TypeId>(in_dtype);
      bool out_int32 = false;
      if (in_dtype_id == kNumberTypeInt32) {
        out_int32 = true;
      }

      LAUNCH_ACLNN(aclnnSearchSorted, device_context, op->stream_id(), sorted_sequence, values, out_int32, in_right,
                   sorter, outputs[0]);
      MS_LOG(DEBUG) << "Run device task SearchSorted end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
