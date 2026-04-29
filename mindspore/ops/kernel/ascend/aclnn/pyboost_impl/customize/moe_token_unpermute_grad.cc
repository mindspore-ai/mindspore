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

#include "kernel/ascend/aclnn/pyboost_impl/customize/moe_token_unpermute_grad.h"
#include <cstdint>
#include <vector>
#include "include/utils/utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr> MoeTokenUnpermuteGradAscendCustomize(
  const TensorPtr &permuted_tokens, const TensorPtr &unpermuted_tokens_grad, const TensorPtr &sorted_indices,
  const std::optional<TensorPtr> &probs, const BoolImmPtr &padded_mode,
  const std::optional<ValueTuplePtr> &restore_shape) {
  auto outputs = InferOutput<2>(prim::kPrimMoeTokenUnpermuteGrad, permuted_tokens, unpermuted_tokens_grad,
                                sorted_indices, probs, padded_mode, restore_shape);

  // Convert ValuePtr to c++ scalar
  auto padded_mode_imm = GetValue<bool>(padded_mode);
  auto input_shape = permuted_tokens->shape();
  auto input_size = SizeToLong(input_shape.size());
  auto tokensNum = input_size > 0 ? input_shape[0] : 0;
  auto hiddenSize = input_size > 1 ? input_shape[1] : 0;

  // Convert ValueTuple to std::vector
  std::vector<int64_t> restore_shape_val = {1, 1};
  if (restore_shape.has_value()) {
    restore_shape_val = ConvertValueTupleToVector<int64_t>(restore_shape.value());
  }

  TensorToDevice(permuted_tokens, unpermuted_tokens_grad, sorted_indices, probs);
  TensorAllocate(outputs);
  if (tokensNum != 0 && hiddenSize != 0) {
    DISPATCH_ACLNN(aclnnMoeTokenUnpermuteGrad, permuted_tokens, unpermuted_tokens_grad, sorted_indices, probs,
                   padded_mode_imm, restore_shape_val, std::get<0>(outputs), std::get<1>(outputs));
  } else {
    MS_LOG(DEBUG) << "For [MoeTokenUnpermuteGrad], input is empty. Do not launch kernel.";
  }
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
