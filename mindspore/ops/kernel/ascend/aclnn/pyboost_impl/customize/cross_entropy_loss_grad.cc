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

#include "kernel/ascend/aclnn/pyboost_impl/customize/cross_entropy_loss_grad.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <tuple>
#include "include/utils/utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "include/pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr CrossEntropyLossGradAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &grad_loss, const TensorPtr &log_prob, const TensorPtr &target,
  const std::optional<TensorPtr> &weight, const std::optional<TensorPtr> &grad_zloss,
  const std::optional<TensorPtr> &lse_for_zloss, const Int64ImmPtr &reduction, const Int64ImmPtr &ignore_index,
  const FP32ImmPtr &label_smoothing, const FP32ImmPtr &lse_square_scale_for_zloss) {
  OpRunner::InferOpOutput(op, grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, reduction, ignore_index,
                          label_smoothing, lse_square_scale_for_zloss);
  auto reduction_imm = static_cast<Reduction>(GetValue<int64_t>(reduction));
  // transform reduction enum value to corresponding value
  auto reduction_value = ops::ConvertReductionStrForAclnn(reduction_imm);
  const auto ignore_index_imm = GetValue<int64_t>(ignore_index);
  const auto label_smoothing_imm = static_cast<double>(label_smoothing->value());
  if (label_smoothing_imm != 0.0) {
    MS_LOG(EXCEPTION)
      << "For 'CrossEntropyLossGrad', 'label_smoothing' must be 0.0 or the calculated result is undefined.";
  }
  const auto lse_square_scale_for_zloss_imm = static_cast<double>(lse_square_scale_for_zloss->value());
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), grad_loss, log_prob, target, weight, grad_zloss,
                                lse_for_zloss);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  TensorPtr target_contiguous = target;
  if (!target->is_contiguous()) {
    MS_LOG(DEBUG) << "For CrossEntropyLossGrad, 'target' is not contiguous.";
    target_contiguous = contiguous(target);
  }
  TensorPtr weight_contiguous = nullptr;
  if (weight.has_value()) {
    auto weight_tensor = weight.value();
    if (!weight_tensor->is_contiguous()) {
      MS_LOG(DEBUG) << "For CrossEntropyLossGrad, 'weight' is not contiguous.";
      weight_contiguous = contiguous(weight_tensor);
    } else {
      weight_contiguous = weight_tensor;
    }
  }

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, grad_loss, log_prob, target_contiguous, weight_contiguous, grad_zloss, lse_for_zloss, reduction_value,
     ignore_index_imm, label_smoothing_imm, lse_square_scale_for_zloss_imm]() {
      MS_LOG(DEBUG) << "Run device task CrossEntropyLossGrad start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();

      std::optional<TensorPtr> weight_opt;
      if (weight_contiguous) {
        weight_opt = weight_contiguous;
      }
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(op->device_context(), grad_loss, log_prob, target_contiguous, weight_opt, grad_zloss,
                                   lse_for_zloss);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(op->device_context(), outputs);
      LAUNCH_ACLNN(aclnnCrossEntropyLossGrad, device_context, op->stream_id(), grad_loss, log_prob, target_contiguous,
                   weight_opt, grad_zloss, lse_for_zloss, reduction_value, ignore_index_imm, label_smoothing_imm,
                   lse_square_scale_for_zloss_imm, outputs[kIndex0]);
      MS_LOG(DEBUG) << "Run device task CrossEntropyLossGrad end";
    }));
  return op->outputs()[kIndex0];
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
