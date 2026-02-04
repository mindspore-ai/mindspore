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

#include "kernel/ascend/aclnn/pyboost_impl/customize/layer_norm_grad_ext.h"
#include <algorithm>
#include <memory>
#include <functional>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void LayerNormGradExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &dy_tensor,
                                     const TensorPtr &x_tensor, const ValueTuplePtr &normalized_shape,
                                     const TensorPtr &mean_tensor, const TensorPtr &variance_tensor,
                                     const TensorPtr &gamma_tensor, const TensorPtr &beta_tensor,
                                     const ValueTuplePtr &output_mask) {
  MS_LOG(DEBUG) << "Call start";
  // Convert ValuePtr to c++ scalr
  OpRunner::InferOpOutput(op, dy_tensor, x_tensor, normalized_shape, mean_tensor, variance_tensor, gamma_tensor,
                          beta_tensor, output_mask);

  std::vector<int64_t> normalized_shape_vector = ConvertValueTupleToVector<int64_t>(normalized_shape);
  std::vector<int64_t> output_mask_vector = ConvertValueTupleToVector<int64_t>(output_mask);
  std::vector<uint8_t> output_mask_u8_vec;
  std::transform(output_mask_vector.begin(), output_mask_vector.end(), std::back_inserter(output_mask_u8_vec),
                 [](const int64_t &value) { return static_cast<uint8_t>(value); });

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), dy_tensor, x_tensor, mean_tensor,
                                variance_tensor, gamma_tensor, beta_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, dy_tensor, x_tensor, normalized_shape_vector, mean_tensor,
                                                  variance_tensor, gamma_tensor, beta_tensor, output_mask_u8_vec]() {
      MS_LOG(DEBUG) << "Run device task LayerNormGradExt start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, dy_tensor, x_tensor, mean_tensor, variance_tensor, gamma_tensor,
                                   beta_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnLayerNormBackward, device_context, op->stream_id(), dy_tensor, x_tensor,
                   normalized_shape_vector, mean_tensor, variance_tensor, gamma_tensor, beta_tensor, output_mask_u8_vec,
                   outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
      MS_LOG(DEBUG) << "Run device task LayerNormGradExt end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
