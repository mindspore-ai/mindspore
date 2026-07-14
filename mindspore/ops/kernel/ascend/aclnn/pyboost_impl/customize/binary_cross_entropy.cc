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

#include "kernel/ascend/aclnn/pyboost_impl/customize/binary_cross_entropy.h"
#include <unordered_map>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "mindapi/base/types.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr BinaryCrossEntropyAscendCustomize(const TensorPtr &input_tensor, const TensorPtr &target_tensor,
                                                    const std::optional<TensorPtr> &weight_tensor,
                                                    const Int64ImmPtr &reduction) {
  MS_LOG(DEBUG) << "BinaryCrossEntropy call start";
  auto output = InferOutput<1>(prim::kPrimBinaryCrossEntropy, input_tensor, target_tensor, weight_tensor, reduction);

  auto reduction_imm = static_cast<Reduction>(GetValue<int64_t>(reduction));
  // transform reduction enum value to corresponding value
  std::unordered_map<Reduction, int64_t> reduction_map = {
    {Reduction::REDUCTION_SUM, 2}, {Reduction::MEAN, 1}, {Reduction::NONE, 0}};
  auto iter = reduction_map.find(reduction_imm);
  if (iter == reduction_map.end()) {
    MS_LOG(EXCEPTION) << "For BinaryCrossEntropy, the value of reduction is invalid.";
  }
  auto reduction_value = iter->second;

  // aclnnBinaryCrossEntropy does not support low-dim weight broadcast; expand weight to input shape.
  // This aligns with the existing MS pattern in l1_loss_ext.cc / mse_loss_ext.cc.
  auto actual_weight = weight_tensor;
  if (weight_tensor.has_value() && weight_tensor.value()->shape() != input_tensor->shape()) {
    kernel::pyboost::RequireGradGuard require_grad_guard(false);
    const auto &input_shape = input_tensor->shape();
    actual_weight = broadcast_to(weight_tensor.value(), input_shape);
  }

  TensorToDevice(input_tensor, target_tensor, actual_weight);
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnBinaryCrossEntropy, input_tensor, target_tensor, actual_weight, reduction_value, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
