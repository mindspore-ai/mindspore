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

#include "plugin/device/ascend/kernel/pyboost/customize/non_zero_ext.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/customize/op_common.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "runtime/device/device_address_utils.h"
#include "kernel/pyboost/op_runner.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/auto_generate/non_zero.h"
#include "mindspore/core/ops/view/unstack_strides_calc.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::vector<tensor::BaseTensorPtr> NonZeroExtAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                             const BaseTensorPtr &input_tensor) {
  MS_LOG(DEBUG) << "NonZeroExt call start";
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto nonzero_op = CREATE_PYBOOST_OP(NonZero, kAscendDevice);
  const auto x_tensor = nonzero_op->Call(input_tensor);
  std::vector<ValuePtr> inputs_unstack;
  inputs_unstack.push_back(x_tensor);
  int64_t axis_data = 1;
  auto input_axis = MakeValue(axis_data);
  auto prim = std::make_shared<Primitive>("Unstack");
  prim->AddAttr(ops::kAxis, input_axis);
  auto storage_info_list = ops::UnstackCalc(prim, inputs_unstack);
  if (!storage_info_list.empty()) {
    std::vector<tensor::BaseTensorPtr> outputs;
    PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x_tensor);
    PyBoostUtils::CreateOutputTensor(op->device_context(), x_tensor, storage_info_list, &outputs);
    op->set_outputs(outputs);

    // Sync
    PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, x_tensor]() {
      auto device_context = op->device_context();
      PyBoostUtils::MallocOpInputs(device_context, x_tensor);
    }));

    op->SetOutputTupleAbstract();
  } else {
    MS_LOG_EXCEPTION << "View unsupported:" << prim->name() << " or input ERROR";
  }
  MS_LOG(DEBUG) << "NonZeroExt call end";
  return op->outputs();
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
