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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inner_non_zero.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/customize/op_common.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/op_runner.h"
#include "include/runtime/pipeline/pipeline.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InnerNonZeroAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor) {
  MS_LOG(DEBUG) << "InnerNonZero Ascend start";
  OpRunner::InferOpOutput(op, input_tensor);

  // set address
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor]() {
    auto device_context = op->device_context();
    // exec aclnnNonzero
    const auto &outputs = op->outputs();

    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    const auto &all_acl_tensor =
      LAUNCH_ACLNN_SYNC(aclnnNonzeroV2, device_context, op->stream_id(), input_tensor, outputs[0]);

    // update shape
    auto output_real_shape = all_acl_tensor[kIndex1];
    // when case:1D-0D tensor,needs to update shape {-1,1} to {0,1}
    if (output_real_shape[0] == -1) {
      output_real_shape[0] = 0;
    }
    auto simple_infer_ptr = op->output_value_simple_info();
    simple_infer_ptr->shape_vector_ = ShapeArray{output_real_shape};
    op->UpdateOutputShape(outputs[kIndex0], output_real_shape);
  }));
  runtime::Pipeline::Get().backend_stage()->Wait();
  MS_LOG(DEBUG) << "InnerNonZero Ascend end";
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
