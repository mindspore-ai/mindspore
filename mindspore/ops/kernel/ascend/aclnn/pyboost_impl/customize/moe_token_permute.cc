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

#include "kernel/ascend/aclnn/pyboost_impl/customize/moe_token_permute.h"
#include <cstdint>
#include <vector>
#include <tuple>
#include "include/utils/utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr> MoeTokenPermuteAscendCustomize(
  const TensorPtr &tokens, const TensorPtr &indices, const std::optional<Int64ImmPtr> &num_out_tokens,
  const BoolImmPtr &padded_mode) {
  auto outputs = InferOutput<2>(prim::kPrimMoeTokenPermute, tokens, indices, num_out_tokens, padded_mode);
  // Convert ValuePtr to c++ scalar
  auto padded_mode_imm = GetValue<bool>(padded_mode);
  auto num_out_tokens_imm = num_out_tokens.has_value() ? GetValue<int64_t>(num_out_tokens.value()) : 0;
  TensorToDevice(tokens, indices);
  TensorAllocate(outputs);
  DISPATCH_ACLNN(aclnnMoeTokenPermute, tokens, indices, num_out_tokens_imm, padded_mode_imm, std::get<0>(outputs),
                 std::get<1>(outputs));
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
