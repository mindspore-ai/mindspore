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

#include "mindspore/ops/kernel/cpu/pyboost/customize/binary_cross_entropy_with_logits.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/cast.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/op_runner.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void BinaryCrossEntropyWithLogitsCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                              const TensorPtr &target_tensor,
                                              const std::optional<TensorPtr> &weight_tensor,
                                              const std::optional<TensorPtr> &posWeight_tensor,
                                              const Int64ImmPtr &reduction) {
  MS_EXCEPTION_IF_NULL(op);
  OpRunner::InferOpOutput(op, input_tensor, target_tensor, weight_tensor, posWeight_tensor, reduction);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, target_tensor, weight_tensor,
                                posWeight_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  const auto &real_input_tensor = PyBoostUtils::CastTensor(input_tensor, kNumberTypeFloat32, device::DeviceType::kCPU);
  const auto &real_target_tensor =
    PyBoostUtils::CastTensor(target_tensor, kNumberTypeFloat32, device::DeviceType::kCPU);
  const auto &real_weight_tensor =
    PyBoostUtils::CastTensor(weight_tensor, kNumberTypeFloat32, device::DeviceType::kCPU);
  const auto &real_posWeight_tensor =
    PyBoostUtils::CastTensor(posWeight_tensor, kNumberTypeFloat32, device::DeviceType::kCPU);

  const auto &outputs = op->outputs();
  const auto &real_outputs = PyBoostUtils::CastTensor(outputs[0], kNumberTypeFloat32, device::DeviceType::kCPU);

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, real_input_tensor, real_target_tensor, real_weight_tensor, real_posWeight_tensor, reduction, real_outputs]() {
      MS_LOG(DEBUG) << "For 'binary_cross_entropy_with_logits', the cpu task start";
      auto device_context = op->device_context();
      PyBoostUtils::MallocOpInputs(device_context, real_input_tensor, real_target_tensor, real_weight_tensor,
                                   real_posWeight_tensor);
      PyBoostUtils::MallocOpOutputs(device_context, {real_outputs});

      const auto &input_address_info =
        PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), op->input_abs(), real_input_tensor,
                                     real_target_tensor, real_weight_tensor, real_posWeight_tensor, reduction);
      const auto &output_address_info =
        PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, real_outputs);

      PyBoostUtils::LaunchKernel(op->primitive(), device_context, input_address_info, output_address_info);
      MS_LOG(DEBUG) << "For 'binary_cross_entropy_with_logits', the cpu task end";
    }));

  const auto &real_output_tensor =
    PyBoostUtils::CastTensor(real_outputs, outputs[0]->data_type(), device::DeviceType::kCPU);

  op->set_outputs({real_output_tensor});
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
