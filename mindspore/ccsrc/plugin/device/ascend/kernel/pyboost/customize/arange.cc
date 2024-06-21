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

#include "plugin/device/ascend/kernel/pyboost/customize/arange.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "runtime/device/device_address_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr ArangeAscendCustomize(const std::shared_ptr<OpRunner> &op, const ScalarPtr &start,
                                            const ScalarPtr &end, const ScalarPtr &step,
                                            const std::optional<Int64ImmPtr> &dtype) {
  OpRunner::InferOpOutput(op, start, end, step, dtype);

  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, start, end, step]() {
    auto device_context = op->device_context();

    PyBoostUtils::MallocOpOutputs(device_context, op->outputs());

    MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
    LAUNCH_ACLNN(aclnnArange, device_context, op->stream_id(), start, end, step, op->output(0));
    MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
