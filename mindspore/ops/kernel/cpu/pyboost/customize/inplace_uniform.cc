/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "mindspore/ops/kernel/cpu/pyboost/customize/inplace_uniform.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InplaceUniformCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                             const ScalarPtr &from, const ScalarPtr &to, const TensorPtr &seed,
                                             const TensorPtr &offset) {
  OpRunner::InferOpOutput(op, input_tensor, from, to, seed, offset);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, seed, offset);
  op->set_outputs({input_tensor});
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, from, to, seed, offset]() {
    auto device_context = op->device_context();
    PyBoostUtils::MallocOpInputs(device_context, input_tensor, seed, offset);
    const auto &input_address_info = PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), op->input_abs(),
                                                                  input_tensor, from, to, seed, offset);
    const auto &output_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, op->outputs());
    PyBoostUtils::LaunchKernel(op->primitive(), device_context, input_address_info, output_address_info);
  }));
  return input_tensor;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
