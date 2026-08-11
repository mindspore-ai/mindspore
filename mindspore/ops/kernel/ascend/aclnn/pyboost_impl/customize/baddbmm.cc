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

#include "kernel/ascend/aclnn/pyboost_impl/customize/baddbmm.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr BaddbmmAscendCustomize(const TensorPtr &input_tensor, const TensorPtr &batch1_tensor,
                                         const TensorPtr &batch2_tensor, const ScalarPtr &beta,
                                         const ScalarPtr &alpha) {
  auto output = InferOutput<1>(prim::kPrimBaddbmm, input_tensor, batch1_tensor, batch2_tensor, beta, alpha);

  // No need to convert input
  TensorToDevice(input_tensor, batch1_tensor, batch2_tensor);
  TensorMalloc(output);

  auto cube_math_type = GetCubeMathTypeForAddOps(IsAllowMatmulHF32());
  DISPATCH_ACLNN(aclnnBaddbmm, input_tensor, batch1_tensor, batch2_tensor, beta, alpha, output, cube_math_type);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
