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

#include "include/pynative/utils/pyboost/customize/gather_nd_ext.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <vector>

#include "mindspore/ops/ops_utils/op_constants.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/math_ops.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void GatherNdCall(const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor, const TensorPtr &y_tensor) {
  MS_EXCEPTION_IF_NULL(op);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x_tensor, y_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, x_tensor, y_tensor]() {
    MS_LOG(DEBUG) << "Run device task GatherNd start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();

    PyBoostUtils::MallocOpInputs(device_context, x_tensor, y_tensor);
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    std::vector<AbstractBasePtr> input_abs{x_tensor->ToAbstract(), y_tensor->ToAbstract()};
    const auto &input_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), input_abs, x_tensor, y_tensor);
    const auto &output_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, outputs);

    const auto primitive = std::make_shared<Primitive>("GatherNd");
    PyBoostUtils::LaunchKernel(primitive, device_context, input_address_info, output_address_info, op->stream_id());
    MS_LOG(DEBUG) << "Run device task GatherNd end";
  }));
}
}  // namespace

tensor::TensorPtr GatherNdExtCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                       const TensorPtr &indices_tensor) {
  OpRunner::InferOpOutput(op, input_tensor, indices_tensor);
  GatherNdCall(op, input_tensor, indices_tensor);
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
