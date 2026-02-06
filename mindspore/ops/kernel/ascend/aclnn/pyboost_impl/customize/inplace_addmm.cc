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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_addmm.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "mindspore/ops/ops_utils/memory_overlap.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InplaceAddmmAscendCustomize(const tensor::TensorPtr &input_tensor,
                                              const tensor::TensorPtr &mat1_tensor,
                                              const tensor::TensorPtr &mat2_tensor, const ScalarPtr &beta,
                                              const ScalarPtr &alpha) {
  InferOutput<1>(prim::kPrimInplaceAddmm, input_tensor, mat1_tensor, mat2_tensor, beta, alpha);

  // No need to convert input
  TensorToDevice(input_tensor, mat1_tensor, mat2_tensor);
  // Check Memory Partial Overlap
  mindspore::pyboost::CheckMemoryOverlap({input_tensor, mat1_tensor, mat2_tensor}, {input_tensor});
  // cubeMathType: 0 - KEEP_DTYPE, 1 - ALLOW_FP32_DOWN_PRECISION
  auto cube_math_type = GetCubeMathType(IsAllowMatmulHF32());
  DISPATCH_ACLNN(aclnnInplaceAddmm, input_tensor, mat1_tensor, mat2_tensor, beta, alpha, cube_math_type);
  return input_tensor;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
