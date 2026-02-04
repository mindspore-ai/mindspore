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

#include "kernel/gpu/pyboost/customize/inner_comm_irecv.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/customize/op_common.h"

#include "ir/tensor_new.h"
namespace mindspore {
namespace kernel {
namespace pyboost {
void InnerCommIrecvGPUCustomize(const std::shared_ptr<OpRunner> &op, const Int64ImmPtr &tag, const Int64ImmPtr &src,
                                const ValueTuplePtr &shape, const StringImmPtr &group, const Int64ImmPtr &dtype) {
  // Create Fake tensor for irecv
  auto shape_vector = ConvertValueTupleToVector<int64_t>(shape);
  auto dtype_id = GetValue<int64_t>(dtype);
  auto input_tensor = tensor::from_spec(static_cast<TypeId>(dtype_id), shape_vector, device::DeviceType::kNone);

  auto pre_func = [op, input_tensor]() {
    OpRunner::InferOpOutput(op, input_tensor);
    // Create device address for output tensors
    PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  };

  auto launch_func = [op]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    const auto &output_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, outputs);

    // Launch kernel
    PyBoostUtils::LaunchKernel(op->primitive(), op->device_context(), output_address_info, output_address_info,
                               op->stream_id(), true);
  };

  CommonCommFunc(op, input_tensor, pre_func, launch_func);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
