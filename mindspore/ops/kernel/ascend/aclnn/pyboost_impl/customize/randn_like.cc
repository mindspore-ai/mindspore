/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/randn_like.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr RandnLikeAscendCustomize(const tensor::TensorPtr &tensor_tensor, const tensor::TensorPtr &seed,
                                           const tensor::TensorPtr &offset, const std::optional<Int64ImmPtr> &dtype,
                                           const std::optional<Int64ImmPtr> &device) {
  auto output = InferOutput<1>(prim::kPrimRandnLike, tensor_tensor, seed, offset, dtype, device);

  auto [seed_imm, offset_imm] = GetGeneratorState(seed, offset);

  TensorToDevice(tensor_tensor);
  TensorMalloc(output);

  constexpr float mean_imm = 0.0;
  constexpr float std_imm = 1.0;
  DISPATCH_ACLNN(aclnnInplaceNormal, output, mean_imm, std_imm, seed_imm, offset_imm);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
