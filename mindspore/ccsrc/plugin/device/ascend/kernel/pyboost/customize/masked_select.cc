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

#include "plugin/device/ascend/kernel/pyboost/customize/masked_select.h"

#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

tensor::BaseTensorPtr MaskedSelectAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                  const BaseTensorPtr &input_tensor, const BaseTensorPtr &mask_tensor) {
  MS_LOG(DEBUG) << op->primitive()->name() << " call start";
  auto device_context = op->device_context();
  auto stream_id = op->stream_id();
  OpRunner::InferOpOutput(op, input_tensor, mask_tensor);
  PyBoostUtils::PrepareOpInputs(device_context, stream_id, input_tensor, mask_tensor);
  PyBoostUtils::PrepareOpOutputs(device_context, stream_id, op->outputs());
  runtime::OpExecutor::GetInstance().WaitAll();
  const auto &outputs = op->outputs();
  // Malloc for input tensors
  PyBoostUtils::MallocOpInputs(device_context, input_tensor, mask_tensor);
  // Malloc for output tensors
  PyBoostUtils::MallocOpOutputs(device_context, outputs);
  auto return_value =
    LAUNCH_ACLNN_SYNC(aclnnMaskedSelect, device_context, op->stream_id(), input_tensor, mask_tensor, outputs[0]);
  auto &all_acl_tensor = std::get<2>(return_value);
  auto output_real_shape = transform::UpdateOutputShape(all_acl_tensor.get<2>());
  outputs[0]->set_shape(output_real_shape);
  op->set_output_abs(outputs[0]->ToAbstract());
  MS_LOG(DEBUG) << "Run device task " << op->primitive()->name() << " end";
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
