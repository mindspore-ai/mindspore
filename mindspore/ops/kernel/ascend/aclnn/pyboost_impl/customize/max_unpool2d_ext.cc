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

#include "kernel/ascend/aclnn/pyboost_impl/customize/max_unpool2d_ext.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/sum_ext.h"
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/ne_scalar.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

tensor::TensorPtr MaxUnpool2DExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                                const TensorPtr &indices,
                                                const std::optional<ValueTuplePtr> &kernel_size,
                                                const std::optional<ValueTuplePtr> &stride,
                                                const std::optional<ValueTuplePtr> &padding,
                                                const std::optional<ValueTuplePtr> &ouput_size) {
  OpRunner::InferOpOutput(op, input_tensor, indices, kernel_size, stride, padding, ouput_size);
  std::vector<int64_t> output_size_vector{};
  std::vector<int64_t> shape{op->output(0)->shape()};
  size_t size = shape.size();

  output_size_vector.push_back(shape[size - kDim2]);
  output_size_vector.push_back(shape[size - kDim1]);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, indices);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, indices, output_size_vector]() {
      MS_LOG(DEBUG) << "Launch MaxUnpool2d start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors

      PyBoostUtils::MallocOpInputs(device_context, input_tensor, indices);

      // Inplace output need be front
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      LAUNCH_ACLNN(aclnnMaxUnpool2d, device_context, op->stream_id(), input_tensor, indices, output_size_vector,
                   outputs[0]);
      MS_LOG(DEBUG) << "Launch MaxUnpool2d end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
