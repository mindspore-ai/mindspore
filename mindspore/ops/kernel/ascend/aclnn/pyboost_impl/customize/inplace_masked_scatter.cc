/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_masked_scatter.h"
#include <memory>
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "mindspore/ops/ops_utils/memory_overlap.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InplaceMaskedScatterAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input,
                                                      const TensorPtr &mask, const TensorPtr &source) {
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input, mask, source);
  op->set_outputs({input});
  // source
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input, mask, source]() {
    auto device_context = op->device_context();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input, mask, source);
    // Check Memory Partial Overlap
    CheckMemory({input, mask, source}, {input});
    // Inplace output need be front
    MS_LOG(DEBUG) << "Call InplaceMaskedScatter start";
    LAUNCH_ACLNN(aclnnInplaceMaskedScatter, device_context, op->stream_id(), input, mask, source);
    MS_LOG(DEBUG) << "Launch InplaceMaskedScatter end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
