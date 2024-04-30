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

#include "plugin/device/ascend/kernel/pyboost/customize/layer_norm_ext.h"
#include <memory>
#include <functional>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void LayerNormExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                 const ValueTuplePtr &normalized_shape,
                                 const std::optional<BaseTensorPtr> &weight_opt_tensor,
                                 const std::optional<BaseTensorPtr> &bias_opt_tensor, const FP32ImmPtr &eps) {
  MS_LOG(DEBUG) << "Call start";

  // Convert ValuePtr to c++ scalr
  OpRunner::InferOpOutput(op, input_tensor, normalized_shape, weight_opt_tensor, bias_opt_tensor, eps);

  std::vector<int64_t> normalized_shape_vector = ConvertValueTupleToVector<int64_t>(normalized_shape);
  auto eps_imm = static_cast<double>(GetValue<float>(eps));

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, weight_opt_tensor,
                                bias_opt_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, input_tensor, normalized_shape_vector, weight_opt_tensor, bias_opt_tensor, eps_imm]() {
      MS_LOG(DEBUG) << "Run device task Add start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor, weight_opt_tensor, bias_opt_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnLayerNorm, device_context, op->stream_id(), input_tensor, normalized_shape_vector,
                   weight_opt_tensor, bias_opt_tensor, eps_imm, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
      MS_LOG(DEBUG) << "Run device task Add end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
