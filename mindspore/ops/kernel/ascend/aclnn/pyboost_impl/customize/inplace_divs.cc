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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_divs.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InplaceDivsAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                             const ScalarPtr &other) {
  MS_LOG(DEBUG) << "Call InplaceDivsAscendCustomize start";
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  op->set_outputs({input_tensor});
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, other]() {
    MS_LOG(DEBUG) << "Run device task InplaceDivs start";
    auto device_context = op->device_context();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor);

    LAUNCH_ACLNN(aclnnInplaceDivs, device_context, op->stream_id(), input_tensor, other);
    MS_LOG(DEBUG) << "Run device task InplaceDivs end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
