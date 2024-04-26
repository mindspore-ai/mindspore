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

#include "plugin/device/ascend/kernel/pyboost/customize/layer_norm_grad_ext.h"
#include <memory>
#include <functional>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void LayerNormGradExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &dy_tensor,
                                     const BaseTensorPtr &x_tensor, const ValueTuplePtr &normalized_shape,
                                     const BaseTensorPtr &mean_tensor, const BaseTensorPtr &variance_tensor,
                                     const BaseTensorPtr &gamma_tensor, const BaseTensorPtr &beta_tensor) {
  MS_LOG(DEBUG) << "Call start";
  // Convert ValuePtr to c++ scalr
  OpRunner::InferOpOutput(op, dy_tensor, x_tensor, normalized_shape, mean_tensor, variance_tensor, gamma_tensor,
                          beta_tensor);

  std::vector<int64_t> normalized_shape_vector = ConvertValueTupleToVector<int64_t>(normalized_shape);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), dy_tensor, x_tensor, mean_tensor,
                                variance_tensor, gamma_tensor, beta_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, dy_tensor, x_tensor, normalized_shape_vector, mean_tensor, variance_tensor, gamma_tensor, beta_tensor]() {
      MS_LOG(DEBUG) << "Run device task Add start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      std::vector<uint8_t> output_mask{1, 1, 1};
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, dy_tensor, x_tensor, mean_tensor, variance_tensor, gamma_tensor,
                                   beta_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnLayerNormBackward, device_context, op->stream_id(), dy_tensor, x_tensor,
                   normalized_shape_vector, mean_tensor, variance_tensor, gamma_tensor, beta_tensor, output_mask,
                   outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
      MS_LOG(DEBUG) << "Run device task Add end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
