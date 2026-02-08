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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_sub_ext.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "mindspore/ops/ops_utils/memory_overlap.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

tensor::TensorPtr InplaceSubExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                               const TensorPtr &other_tensor, const ScalarPtr &alpha) {
  MS_LOG(DEBUG) << "Call InplaceSubExt start";
  if ((PyBoostUtils::IsBool(alpha) && (!PyBoostUtils::IsBool(input_tensor) || !PyBoostUtils::IsBool(other_tensor)))) {
    MS_EXCEPTION(TypeError) << "For " << op->primitive()->name()
                            << ", Boolean alpha only supported for Boolean results.";
  }
  if ((PyBoostUtils::IsFloat(alpha) &&
       (!PyBoostUtils::IsFloat(input_tensor) || !PyBoostUtils::IsFloat(other_tensor)))) {
    MS_EXCEPTION(TypeError) << "For " << op->primitive()->name()
                            << ", For integral input tensors, argument alpha must not be a floating point number";
  }
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, other_tensor);
  op->set_outputs({input_tensor});
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, other_tensor, alpha]() {
    MS_LOG(DEBUG) << "Run device task InplaceSubExt start";
    auto device_context = op->device_context();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor, other_tensor);
    // Check Memory Partial Overlap
    CheckMemory({input_tensor, other_tensor}, {input_tensor});
    // Inplace output need be front
    LAUNCH_ACLNN(aclnnInplaceSub, device_context, op->stream_id(), input_tensor, other_tensor, alpha);
    MS_LOG(DEBUG) << "Launch InplaceSubExt end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
