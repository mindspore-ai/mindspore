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

#include "mindspore/ops/kernel/cpu/pyboost/customize/min.h"
#include "mindspore/ops/kernel/cpu/pyboost/customize/max.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/min.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/max.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/cast.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void MinOrMaxCPUCall(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor, const std::string &reduce_op) {
  MS_EXCEPTION_IF_NULL(op);
  OpRunner::InferOpOutput(op, input_tensor);

  if (input_tensor->data_type() == kNumberTypeFloat16) {
    // Increase the precision to float32 for calculation
    const auto &cast_input_tensor =
      PyBoostUtils::CastTensor(input_tensor, kNumberTypeFloat32, device::DeviceType::kCPU);
    tensor::TensorPtr cast_output_tensor;
    if (reduce_op == prim::kPrimReduceMin->name()) {
      const auto &min_op = CREATE_PYBOOST_OP(Min, device::DeviceType::kCPU);
      cast_output_tensor = min_op->Call(cast_input_tensor);
    } else {
      const auto &max_op = CREATE_PYBOOST_OP(Max, device::DeviceType::kCPU);
      cast_output_tensor = max_op->Call(cast_input_tensor);
    }
    // After calculation, reduce the precision to float16
    const auto &output_tensor =
      PyBoostUtils::CastTensor(cast_output_tensor, kNumberTypeFloat16, device::DeviceType::kCPU);
    op->set_outputs({output_tensor});
  } else {
    auto axis = MakeValue<std::vector<int64_t>>({});
    auto keep_dims = MakeValue<bool>(false);
    std::vector<AbstractBasePtr> input_abs{input_tensor->ToAbstract(), axis->ToAbstract(), keep_dims->ToAbstract()};

    PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
    PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

    PyBoostUtils::DispatchRun(
      std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, axis, keep_dims, input_abs, reduce_op]() {
        MS_LOG(DEBUG) << "For '" << op->primitive()->name() << "', the cpu task start";
        auto device_context = op->device_context();
        const auto &outputs = op->outputs();
        const auto primitive = std::make_shared<Primitive>(reduce_op);
        MS_EXCEPTION_IF_NULL(primitive);

        PyBoostUtils::MallocOpInputs(device_context, input_tensor);
        PyBoostUtils::MallocOpOutputs(device_context, outputs);

        const auto &input_address_info =
          PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), input_abs, input_tensor, axis, keep_dims);
        const auto &output_address_info =
          PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, outputs);

        PyBoostUtils::LaunchKernel(primitive, device_context, input_address_info, output_address_info);
        MS_LOG(DEBUG) << "For '" << op->primitive()->name() << "', the cpu task end";
      }));
  }
}
}  // namespace

void MinCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor) {
  MinOrMaxCPUCall(op, input_tensor, prim::kPrimReduceMin->name());
}

void MaxCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor) {
  MinOrMaxCPUCall(op, input_tensor, prim::kPrimReduceMax->name());
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
