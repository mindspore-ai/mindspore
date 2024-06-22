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

#include "plugin/device/ascend/kernel/pyboost/customize/non_zero.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/customize/op_common.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "runtime/device/device_address_utils.h"
#include "kernel/pyboost/op_runner.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr NonZeroAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor) {
  MS_LOG(DEBUG) << "NonZero Ascend start";
  OpRunner::InferOpOutput(op, input_tensor);

  auto device_context = op->device_context();

  // set address
  PyBoostUtils::PrepareOpInputs(device_context, op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  runtime::OpExecutor::GetInstance().WaitAll();

  // exec aclnnNonzero
  const auto &outputs = op->outputs();
  // Malloc for input tensors
  PyBoostUtils::MallocOpInputs(device_context, input_tensor);
  // Malloc for output tensors
  PyBoostUtils::MallocOpOutputs(device_context, outputs);
  auto return_values = LAUNCH_ACLNN_SYNC(aclnnNonzero, device_context, op->stream_id(), input_tensor, outputs[0]);
  auto &all_acl_tensor = std::get<2>(return_values);

  // update shape
  auto output_real_shape = transform::UpdateOutputShape(all_acl_tensor.get<1>());
  // when case:1D-0D tensor,needs to update shape {-1,1} to {0,1}
  if (output_real_shape[0] == -1) {
    output_real_shape[0] = 0;
  }
  auto simple_infer_ptr = op->output_value_simple_info();
  simple_infer_ptr->shape_vector_ = ShapeArray{output_real_shape};
  outputs[0]->set_shape(output_real_shape);
  MS_LOG(DEBUG) << "NonZero Ascend end";

  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
