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

#include "kernel/ascend/aclnn/pyboost_impl/customize/adaptive_max_pool2d.h"
#include <memory>
#include <tuple>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
constexpr int kShapeDim2d = 2;
}  // namespace
std::tuple<tensor::TensorPtr, tensor::TensorPtr> AdaptiveMaxPool2DAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                                                  const TensorPtr &input_tensor,
                                                                                  const ValueTuplePtr &output_size) {
  OpRunner::InferOpOutput(op, input_tensor, output_size);

  std::vector<int64_t> output_size_vector = ConvertValueTupleToVector<int64_t>(output_size);
  if (output_size_vector.size() != 2) {
    MS_LOG(EXCEPTION) << "For AdaptiveMaxPool2D, the output_size size should be 2.";
  }
  std::vector<int64_t> output_size_update;

  auto input_shape = input_tensor->shape();
  auto input_shape_size = input_shape.size();
  constexpr int kShapeDimNone = -1;
  for (auto i = 0; i < kShapeDim2d; i++) {
    if (output_size_vector[i] != kShapeDimNone) {
      output_size_update.emplace_back(output_size_vector[i]);
    } else {
      output_size_update.emplace_back(input_shape[input_shape_size - kShapeDim2d + i]);
    }
  }

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, output_size_update]() {
    auto device_context = op->device_context();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, op->outputs());

    LAUNCH_ACLNN(aclnnAdaptiveMaxPool2d, device_context, op->stream_id(), input_tensor, output_size_update,
                 op->output(0), op->output(1));
  }));
  return std::make_tuple(op->output(0), op->output(1));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
