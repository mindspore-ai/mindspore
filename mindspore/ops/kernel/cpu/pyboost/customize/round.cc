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

#include "mindspore/ops/kernel/cpu/pyboost/customize/round.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/cast.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/round.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/op_runner.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void RoundCPUCall(const std::shared_ptr<OpRunner> &op, const TensorPtr &input, const ValuePtr &decimals,
                  const std::vector<AbstractBasePtr> &input_abs) {
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input, decimals, input_abs]() {
    MS_LOG(DEBUG) << "The cpu task 'Round' start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();

    PyBoostUtils::MallocOpInputs(device_context, input);
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    const auto &input_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), input_abs, input, decimals);
    const auto &output_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, outputs);

    PyBoostUtils::LaunchKernel(op->primitive(), device_context, input_address_info, output_address_info);
    MS_LOG(DEBUG) << "For 'Round', the cpu task end";
  }));
}
}  // namespace

void RoundCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input, const Int64ImmPtr &decimals) {
  OpRunner::InferOpOutput(op, input, decimals);

  TensorPtr act_tensor = input;

  if (act_tensor->data_type() == kNumberTypeFloat16) {
    // Increase the precision to float32 for calculation
    const auto &cast_input_tensor = PyBoostUtils::CastTensor(act_tensor, kNumberTypeFloat32, device::DeviceType::kCPU);
    const auto &round_op = CREATE_PYBOOST_OP(Round, device::DeviceType::kCPU);
    const auto &cast_output_tensor = round_op->Call(cast_input_tensor, decimals);
    // After calculation, reduce the precision to float16
    const auto &output_tensor =
      PyBoostUtils::CastTensor(cast_output_tensor, kNumberTypeFloat16, device::DeviceType::kCPU);
    op->set_outputs({output_tensor});
  } else {
    std::vector<AbstractBasePtr> new_input_abs{act_tensor->ToAbstract(), decimals->ToAbstract()};
    RoundCPUCall(op, act_tensor, decimals, new_input_abs);
  }
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
