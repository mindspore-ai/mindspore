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

#include "plugin/device/ascend/kernel/pyboost/customize/cumsum_ext.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "runtime/device/device_address_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr CumsumExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                               const Int64ImmPtr &dim, const std::optional<Int64ImmPtr> &dtype) {
  OpRunner::InferOpOutput(op, input_tensor, dim, dtype);

  auto dim_value = GetValue<int64_t>(dim);

  // Infer function has confirmed the actual dtype of output
  auto output_tensor = op->output_abs()->GetType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(output_tensor);
  TypeId out_dtype = output_tensor->element()->type_id();

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, dim_value, out_dtype]() {
    auto device_context = op->device_context();

    PyBoostUtils::MallocOpInputs(device_context, input_tensor);
    PyBoostUtils::MallocOpOutputs(device_context, op->outputs());

    MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
    LAUNCH_ACLNN(aclnnCumsum, device_context, op->stream_id(), input_tensor, dim_value, out_dtype, op->output(0));
    MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
