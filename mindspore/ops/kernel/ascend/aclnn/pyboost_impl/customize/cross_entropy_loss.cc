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

#include "kernel/ascend/aclnn/pyboost_impl/customize/cross_entropy_loss.h"
#include <cstdint>
#include <vector>
#include <tuple>
#include "include/utils/utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> CrossEntropyLossAscendCustomize(
  const TensorPtr &input, const TensorPtr &target, const std::optional<TensorPtr> &weight, const Int64ImmPtr &reduction,
  const Int64ImmPtr &ignore_index, const FP32ImmPtr &label_smoothing, const FP32ImmPtr &lse_square_scale_for_zloss,
  const BoolImmPtr &return_zloss) {
  auto outputs = InferOutput<4>(prim::kPrimCrossEntropyLoss, input, target, weight, reduction, ignore_index,
                                label_smoothing, lse_square_scale_for_zloss, return_zloss);
  auto reduction_imm = static_cast<Reduction>(GetValue<int64_t>(reduction));
  // transform reduction enum value to corresponding value
  const char *reduction_value = ops::ConvertReductionStrForAclnn(reduction_imm);
  const auto ignore_index_imm = GetValue<int64_t>(ignore_index);
  const auto label_smoothing_imm = static_cast<double>(label_smoothing->value());
  const auto lse_square_scale_for_zloss_imm = static_cast<double>(lse_square_scale_for_zloss->value());
  const auto return_zloss_imm = GetValue<bool>(return_zloss);
  TensorToDevice(input, target, weight);
  TensorAllocate(outputs);

  TensorPtr input_contiguous = input;
  if (!input->is_contiguous()) {
    MS_LOG(DEBUG) << "For CrossEntropyLoss, 'input' is not contiguous.";
    input_contiguous = contiguous(input);
  }
  TensorPtr target_contiguous = target;
  if (!target->is_contiguous()) {
    MS_LOG(DEBUG) << "For CrossEntropyLoss, 'target' is not contiguous.";
    target_contiguous = contiguous(target);
  }
  TensorPtr weight_contiguous = nullptr;
  if (weight.has_value()) {
    auto weight_tensor = weight.value();
    if (!weight_tensor->is_contiguous()) {
      MS_LOG(DEBUG) << "For CrossEntropyLoss, 'weight' is not contiguous.";
      weight_contiguous = contiguous(weight_tensor);
    } else {
      weight_contiguous = weight_tensor;
    }
  }

  std::optional<TensorPtr> weight_opt;
  if (weight_contiguous) {
    weight_opt = weight_contiguous;
  }
  const auto &loss = std::get<0>(outputs);
  const auto &log_prob = std::get<1>(outputs);
  const auto &zloss = std::get<2>(outputs);
  const auto &lse_for_zloss = std::get<3>(outputs);
  DISPATCH_ACLNN(aclnnCrossEntropyLoss, input_contiguous, target_contiguous, weight_opt, reduction_value,
                 ignore_index_imm, label_smoothing_imm, lse_square_scale_for_zloss_imm, return_zloss_imm, loss,
                 log_prob, zloss, lse_for_zloss);
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
