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

#include "include/pynative/utils/pyboost/functions/composite/empty.h"
#include <memory>
#include <string>
#include <vector>
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/functions/composite/composite_utils.h"
#include "include/pynative/utils/runtime/op_runner.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "include/runtime/utils/dispatch/dispatch_env.h"
#include "include/pynative/utils/pyboost/functions/dispatch.h"
#include "mindspore/core/include/ops/op_def_utils.h"
#include "mindspore/core/include/utils/check_convert_utils.h"
#include "mindspore/core/include/utils/stream_guard.h"
#include "mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_e.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr empty(const ValueTuplePtr &size, const std::optional<Int64ImmPtr> &dtype,
                        const std::optional<Int64ImmPtr> &device, const BoolImmPtr &pin_memory) {
  MS_LOG(DEBUG) << "In empty function";

  bool skip_tracker = ProfileTracker::ProfileTrackerTask(prim::kPrimEmpty);

  MS_LOG(DEBUG) << "Call EmptyCustomize start";
  ShapeVector output_shape = GetShape(size);
  TypeId data_type = GetDataType(nullptr, dtype);
  auto device_type = GetDeviceName(nullptr, device);

  for (size_t i = 0; i < output_shape.size(); ++i) {
    MS_CHECK_VALUE(output_shape[i] >= 0,
                   CheckAndConvertUtils::FormatCheckIntegerMsg(std::to_string(i) + "th dimension of input shape",
                                                               output_shape[i], kGreaterEqual, 0, prim::kPrimEmpty));
  }

  auto device_ctx = runtime::OpRunner::GetDeviceContext(device_type);
  MS_EXCEPTION_IF_NULL(device_ctx);

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
    MS_LOG(DEBUG) << "Run device task Empty end";
  }));

  ProfileTracker::ProfileTrackerInput(prim::kPrimEmpty, skip_tracker, size, dtype, device, pin_memory);
  ProfileTracker::ProfileTrackerOutput(prim::kPrimEmpty, skip_tracker, outputs[0]);

  return outputs[0];
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
