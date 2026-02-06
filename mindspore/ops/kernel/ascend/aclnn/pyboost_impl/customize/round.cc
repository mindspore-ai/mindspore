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

#include "kernel/ascend/aclnn/pyboost_impl/customize/round.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr RoundAscendCustomize(const TensorPtr &input_tensor, const std::optional<Int64ImmPtr> &decimals) {
  auto output = InferOutput<1>(prim::kPrimRound, input_tensor, decimals);

  int64_t decimals_imm = 0;
  if (decimals.has_value()) {
    decimals_imm = GetValue<int64_t>(decimals.value());
  }

  TensorToDevice(input_tensor);
  TensorAllocate(output);
  DISPATCH_ACLNN(aclnnRoundDecimals, input_tensor, decimals_imm, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
