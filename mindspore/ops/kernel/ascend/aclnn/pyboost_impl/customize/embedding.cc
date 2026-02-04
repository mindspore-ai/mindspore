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

#include "kernel/ascend/aclnn/pyboost_impl/customize/embedding.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/auto_generate/copy.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void InplaceEmbeddingRenormCall(const std::shared_ptr<OpRunner> &op, const TensorPtr &input, const TensorPtr &weight,
                                const std::optional<FP32ImmPtr> &max_norm, const FP32ImmPtr &norm_type) {
  if (!max_norm.has_value()) {
    return;
  }

  kernel::pyboost::RequireGradGuard require_grad_guard(false);
  TensorPtr copy_input = nullptr;
  auto copy_op = CREATE_PYBOOST_OP(Copy, device::DeviceType::kAscend);
  copy_input = copy_op->Call(input);
  copy_input = view(copy_input, {-1});

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), copy_input, weight);
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, copy_input, weight, max_norm, norm_type]() {
      MS_LOG(DEBUG) << "Run device task EmbeddingRenorm start";
      auto device_context = op->device_context();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, copy_input, weight);
      MS_EXCEPTION_IF_NULL(max_norm.value());
      LAUNCH_ACLNN(aclnnEmbeddingRenorm, device_context, op->stream_id(), weight, copy_input,
                   static_cast<double>(max_norm.value()->value()), static_cast<double>(norm_type->value()));
      MS_LOG(DEBUG) << "Run device task EmbeddingRenorm end";
    }));
}
}  // namespace

tensor::TensorPtr EmbeddingAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input,
                                           const TensorPtr &weight, const std::optional<Int64ImmPtr> &padding_idx,
                                           const std::optional<FP32ImmPtr> &max_norm, const FP32ImmPtr &norm_type,
                                           const BoolImmPtr &scale_grad_by_freq) {
  MS_EXCEPTION_IF_NULL(op);

  OpRunner::InferOpOutput(op, input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq);

  InplaceEmbeddingRenormCall(op, input, weight, max_norm, norm_type);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input, weight);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input, weight]() {
    MS_LOG(DEBUG) << "Run device task Embedding start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input, weight);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    LAUNCH_ACLNN(aclnnEmbedding, device_context, op->stream_id(), weight, input, outputs[0]);
    MS_LOG(DEBUG) << "Run device task Embedding end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
