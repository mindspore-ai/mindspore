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

#include "kernel/gpu/pyboost/customize/inner_comm_isend.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/customize/op_common.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void InnerCommIsendGPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                const Int64ImmPtr &dst, const StringImmPtr &group, const Int64ImmPtr &tag) {
  auto pre_func = [op, input_tensor]() {
    OpRunner::InferOpOutput(op, input_tensor);

    PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
    PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  };

  auto launch_func = [op, input_tensor]() {
    auto device_context = op->device_context();

    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor);

    // Get inputs kernel tensors, the not-tensor value will malloc here
    const auto &input_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), op->input_abs(), input_tensor);

    // Launch kernel
    PyBoostUtils::LaunchKernel(op->primitive(), op->device_context(), input_address_info, input_address_info,
                               op->stream_id(), true);
  };

  CommonCommFunc(op, input_tensor, pre_func, launch_func);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
