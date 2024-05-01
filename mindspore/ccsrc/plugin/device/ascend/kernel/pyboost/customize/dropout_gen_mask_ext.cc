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

#include "plugin/device/ascend/kernel/pyboost/customize/dropout_gen_mask_ext.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "runtime/device/device_address_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr DropoutGenMaskExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &shape,
                                                       const FP32ImmPtr &p, const Int64ImmPtr &seed,
                                                       const Int64ImmPtr &offset, const Int64ImmPtr &dtype) {
  OpRunner::InferOpOutput(op, shape, p, seed, offset, dtype);

  auto shape_vector = ConvertValueTupleToVector<int64_t>(shape);
  // Create device address for output tensors.
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, shape_vector, p, seed, offset, dtype]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(op->device_context(), outputs);
    auto p_value = static_cast<double>(p->value());
    auto seed_value = static_cast<int64_t>(seed->value());
    auto offset_value = static_cast<int64_t>(offset->value());
    auto dtype_value = static_cast<TypeId>(dtype->value());
    LAUNCH_ACLNN(aclnnDropoutGenMaskV2, device_context, op->stream_id(), shape_vector, p_value, seed_value,
                 offset_value, dtype_value, outputs[kIndex0]);
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
