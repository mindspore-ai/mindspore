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

#include "include/pynative/utils/pyboost/functions/composite/empty_like.h"
#include <memory>
#include <string>
#include <vector>
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/functions/composite/composite_utils.h"
#include "include/pynative/utils/runtime/op_runner.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "include/runtime/utils/dispatch/dispatch_env.h"
#include "include/pynative/utils/pyboost/functions/dispatch.h"
#include "mindspore/core/include/utils/stream_guard.h"
#include "mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_e.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr empty_like(const tensor::TensorPtr &input_tensor, const std::optional<Int64ImmPtr> &dtype,
                             const std::optional<Int64ImmPtr> &device, const BoolImmPtr &pin_memory) {
  MS_LOG(DEBUG) << "In empty_like function";

  bool skip_tracker = ProfileTracker::ProfileTrackerTask(prim::kPrimEmptyLike);

  MS_LOG(DEBUG) << "Call EmptyLikeCustomize start";
  TypeId data_type = GetDataType(input_tensor, dtype);
  auto device_type = GetDeviceName(input_tensor, device);

  auto device_ctx = runtime::OpRunner::GetDeviceContext(device_type);
  MS_EXCEPTION_IF_NULL(device_ctx);

  auto output_shape = input_tensor->shape();
  std::vector<tensor::TensorPtr> outputs;
  PyBoostUtils::CreateOutputTensor(data_type, output_shape, &outputs);
  PyBoostUtils::PrepareOpOutputs(device_ctx, CurrentStream::id(), outputs);
  if (pin_memory->value()) {
    HandlePinMemory(outputs, device_type);
  }

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([outputs, device_ctx]() {
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_ctx, outputs);
    MS_LOG(DEBUG) << "Run device task EmptyLike end";
  }));

  ProfileTracker::ProfileTrackerInput(prim::kPrimEmptyLike, skip_tracker, input_tensor, dtype, device, pin_memory);
  ProfileTracker::ProfileTrackerOutput(prim::kPrimEmptyLike, skip_tracker, outputs[0]);

  return outputs[0];
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
