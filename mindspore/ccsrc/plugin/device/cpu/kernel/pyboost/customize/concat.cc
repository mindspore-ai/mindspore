/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/pyboost/customize/concat.h"

#include <vector>
#include "ir/scalar.h"
#include "ir/value.h"
#include "ir/tensor.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "mindspore/core/ops/auto_generate/gen_ops_primitive.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void ConcatCpuCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &tensors, const Int64ImmPtr &axis) {
  MS_EXCEPTION_IF_NULL(op);
  OpRunner::InferOpOutput(op, tensors, axis);
  std::vector<tensor::TensorPtr> tensors_vector = ConvertValueTupleToVector<tensor::TensorPtr>(tensors);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), tensors_vector);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Sync
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, tensors_vector, axis]() {
    MS_LOG(DEBUG) << "For 'Concat', the cpu task start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    PyBoostUtils::MallocOpInputs(device_context, tensors_vector);
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    const auto &input_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), op->input_abs(), tensors_vector, axis);
    const auto &output_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, outputs);

    PyBoostUtils::LaunchKernel(op->primitive(), device_context, input_address_info, output_address_info);
    MS_LOG(DEBUG) << "For 'Concat', the cpu task end";
  }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
