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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inner_moe_token_unpermute.h"
#include <cstdint>
#include <vector>
#include "include/utils/utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InnerMoeTokenUnpermuteAscendCustomize(const TensorPtr &permuted_tokens,
                                                        const TensorPtr &sorted_indices,
                                                        const std::optional<TensorPtr> &probs,
                                                        const BoolImmPtr &padded_mode,
                                                        const std::optional<ValueTuplePtr> &restore_shape) {
  auto output = InferOutput<1>(prim::kPrimInnerMoeTokenUnpermute, permuted_tokens, sorted_indices, probs, padded_mode,
                               restore_shape);

  // Convert ValuePtr to c++ scalar
  auto padded_mode_imm = GetValue<bool>(padded_mode);

  // Convert ValueTuple to std::vector
  std::vector<int64_t> restore_shape_val = {1};
  if (restore_shape.has_value()) {
    restore_shape_val = ConvertValueTupleToVector<int64_t>(restore_shape.value());
  }

  TensorToDevice(permuted_tokens, sorted_indices, probs);
  TensorMalloc(output);

  DISPATCH_ACLNN(aclnnMoeTokenUnpermute, permuted_tokens, sorted_indices, probs, padded_mode_imm, restore_shape_val,
                 output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
