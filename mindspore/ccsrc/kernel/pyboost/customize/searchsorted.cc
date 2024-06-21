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

#include "mindspore/ccsrc/kernel/pyboost/customize/searchsorted.h"
#include <memory>
#include <utility>

namespace mindspore {
namespace kernel {
namespace pyboost {

tensor::BaseTensorPtr SearchSortedCustomizeCall(const std::shared_ptr<OpRunner> &op,
                                                const BaseTensorPtr &sorted_sequence, const BaseTensorPtr &values,
                                                const std::optional<BaseTensorPtr> &sorter, const Int64ImmPtr &dtype,
                                                const BoolImmPtr &right) {
  OpRunner::InferOpOutput(op, sorted_sequence, values, sorter, dtype, right);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), sorted_sequence, values, sorter);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, sorted_sequence, values, sorter, dtype, right]() {
      MS_LOG(DEBUG) << "Run device task searchsorted start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();

      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, sorted_sequence, values, sorter);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      // Get inputs kernel tensors, the not-tensor value will malloc here
      const auto &input_address_info = PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), op->input_abs(),
                                                                    sorted_sequence, values, sorter, dtype, right);

      // Get outputs kernel tensors
      const auto &output_address_info =
        PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, outputs);

      PyBoostUtils::LaunchKernel(op->primitive(), op->device_context(), input_address_info, output_address_info,
                                 op->stream_id());
      MS_LOG(DEBUG) << "Run device task searchsorted end";
    }));

  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
