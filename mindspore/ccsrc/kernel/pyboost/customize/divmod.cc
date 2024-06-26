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
#include "kernel/pyboost/pyboost_utils.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/ccsrc/kernel/pyboost/customize/divmod.h"
#include "kernel/pyboost/auto_generate/div.h"
#include "ops/op_enum.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void FloorDivCall(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &x_tensor, const BaseTensorPtr &y_tensor) {
  MS_EXCEPTION_IF_NULL(op);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x_tensor, y_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, x_tensor, y_tensor]() {
    MS_LOG(DEBUG) << "Run device task DivMod-FloorDiv' start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();

    PyBoostUtils::MallocOpInputs(device_context, x_tensor, y_tensor);
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    std::vector<AbstractBasePtr> input_abs{x_tensor->ToAbstract(), y_tensor->ToAbstract()};
    const auto &input_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), input_abs, x_tensor, y_tensor);
    const auto &output_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, outputs);

    const auto primitive = std::make_shared<Primitive>(prim::kPrimFloorDiv->name());
    PyBoostUtils::LaunchKernel(primitive, device_context, input_address_info, output_address_info, op->stream_id());
    MS_LOG(DEBUG) << "Run device task DivMod-FloorDiv end";
  }));
}

void TruncCall(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor) {
  MS_EXCEPTION_IF_NULL(op);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor]() {
    MS_LOG(DEBUG) << "For 'DivMod', the gpu task 'Trunc' start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();

    PyBoostUtils::MallocOpInputs(device_context, input_tensor);
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    std::vector<AbstractBasePtr> input_abs{input_tensor->ToAbstract()};
    const auto &input_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), input_abs, input_tensor);
    const auto &output_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, outputs);

    const auto primitive = std::make_shared<Primitive>(prim::kPrimTrunc->name());
    PyBoostUtils::LaunchKernel(primitive, device_context, input_address_info, output_address_info, op->stream_id());
    MS_LOG(DEBUG) << "Run device task DivMod-Trunc end";
  }));
}
}  // namespace
tensor::BaseTensorPtr DivModCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &x_tensor,
                                      const BaseTensorPtr &y_tensor, const std::optional<Int64ImmPtr> &rounding_mode) {
  OpRunner::InferOpOutput(op, x_tensor, y_tensor, rounding_mode);

  auto mode = 0;
  if (rounding_mode.has_value()) mode = GetValue<int64_t>(rounding_mode.value());

  if (mode == ops::RoundingMode::FLOOR) {
    FloorDivCall(op, x_tensor, y_tensor);
  } else {
    const auto &div_op = CREATE_PYBOOST_OP(Div, op->device_context()->device_context_key_.device_name_);
    div_op->Call(x_tensor, y_tensor);

    if (mode == ops::RoundingMode::TRUNC) {
      auto act_tensor = PyBoostUtils::CastTensor(div_op->outputs()[0], x_tensor->Dtype()->type_id(),
                                                 op->device_context()->device_context_key_.device_name_);
      TruncCall(op, act_tensor);
    } else {
      op->set_outputs(div_op->outputs());
    }
  }

  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
