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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_put.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "mindspore/ops/ops_utils/memory_overlap.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InplacePutAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                            const TensorPtr &index, const TensorPtr &source,
                                            const BoolImmPtr &accumulate) {
  MS_LOG(DEBUG) << "Call aclnnInplacePut start";

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, index, source);
  auto accumulate_imm = GetValue<bool>(accumulate);

  op->set_outputs({input_tensor});
  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, index, source, accumulate_imm]() {
      auto device_context = op->device_context();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor, index, source);
      // Check Memory Partial Overlap
      CheckMemory({input_tensor, index, source}, {input_tensor});
      // Inplace output need be front
      LAUNCH_ACLNN(aclnnInplacePut, device_context, op->stream_id(), input_tensor, index, source, accumulate_imm);
      MS_LOG(DEBUG) << "Launch aclnnInplacePut end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
