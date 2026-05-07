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

#include "kernel/ascend/aclnn/pyboost_impl/customize/randperm_ext.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr RandpermExtAscendCustomize(const Int64ImmPtr &n, const tensor::TensorPtr &seed,
                                             const tensor::TensorPtr &offset, const std::optional<Int64ImmPtr> &dtype) {
  auto output = InferOutput<1>(prim::kPrimRandpermExt, n, seed, offset, dtype);
  auto [seed_imm, offset_imm] = GetGeneratorState(seed, offset);
  auto n_imm = GetValueWithCheck<int64_t>(n);

  TensorMalloc(output);

  DISPATCH_ACLNN(aclnnRandperm, n_imm, seed_imm, offset_imm, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
