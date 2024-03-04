/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/pyboost/customize/tile.h"

#include <memory>
#include <vector>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/py_boost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "runtime/pynative/op_executor.h"
#include "utils/log_adapter.h"

namespace mindspore::kernel::pyboost {
void TileAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_x_tensor,
                         const ValueTuplePtr &dims) {
  MS_EXCEPTION_IF_NULL(op);
  OpRunner::InferOpOutput(op, input_x_tensor, dims);
  std::vector<int64_t> multiples_vector = ConvertValueTupleToVector<int64_t>(dims);

  // Expand dims with 1 in head when its length is less than x rank.
  MS_EXCEPTION_IF_NULL(input_x_tensor);
  auto x_dim = LongToSize(input_x_tensor->DataDim());
  if (x_dim > multiples_vector.size()) {
    multiples_vector.reserve(x_dim);
    auto expand_len = x_dim - multiples_vector.size();
    (void)multiples_vector.insert(multiples_vector.begin(), expand_len, 1);
  }

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_x_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_x_tensor, multiples_vector]() {
    MS_LOG(DEBUG) << "Run device task Tile start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    PyBoostUtils::MallocOpInputs(device_context, input_x_tensor);
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    LAUNCH_ACLNN(aclnnRepeat, device_context, op->stream_id(), input_x_tensor, multiples_vector, outputs[0]);
    MS_LOG(DEBUG) << "Run device task Tile end";
  }));
}
}  // namespace mindspore::kernel::pyboost
