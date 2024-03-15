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

#include "plugin/device/ascend/kernel/pyboost/customize/embedding_dense_backward.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr EmbeddingDenseBackwardAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                        const TensorPtr &grad_tensor, const TensorPtr &indices_tensor,
                                                        const Int64ImmPtr &num_weights,
                                                        const std::optional<Int64ImmPtr> &padding_idx,
                                                        const BoolImmPtr &scale_grad_by_freq) {
  MS_EXCEPTION_IF_NULL(op);

  OpRunner::InferOpOutput(op, grad_tensor, indices_tensor, num_weights, padding_idx, scale_grad_by_freq);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), grad_tensor, indices_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  MS_EXCEPTION_IF_NULL(num_weights);
  auto num_weights_imm = num_weights->value();
  // the type of padding_idx is uint64_t in aclnnEmbeddingDenseBackward api,
  // but the type of padding_idx is int64_t in the operator belowing aclnn api where -1 indicates None value
  // this maybe a risk.
  int64_t padding_idx_imm = 0xFFFFFFFF;

  if (padding_idx.has_value()) {
    MS_EXCEPTION_IF_NULL(padding_idx.value());
    padding_idx_imm = padding_idx.value()->value();
    padding_idx_imm = padding_idx_imm < 0 ? padding_idx_imm + num_weights_imm : padding_idx_imm;
  }

  MS_EXCEPTION_IF_NULL(scale_grad_by_freq);
  auto scale_grad_by_freq_imm = scale_grad_by_freq->value();

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, grad_tensor, indices_tensor, num_weights_imm, padding_idx_imm, scale_grad_by_freq_imm]() {
      MS_LOG(DEBUG) << "Run device task EmbeddingDenseBackward start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, grad_tensor, indices_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnEmbeddingDenseBackward, device_context, op->stream_id(), grad_tensor, indices_tensor,
                   static_cast<uint64_t>(num_weights_imm), static_cast<uint64_t>(padding_idx_imm),
                   scale_grad_by_freq_imm, outputs[0]);

      MS_LOG(DEBUG) << "Run device task EmbeddingDenseBackward end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
