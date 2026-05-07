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

#include "kernel/ascend/aclnn/pyboost_impl/customize/quant_batch_matmul.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "primitive/auto_generate/gen_ops_primitive_q.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr QuantMatmulV4AscendCustomize(const tensor::TensorPtr &x1_tensor, const tensor::TensorPtr &x2_tensor,
                                               const tensor::TensorPtr &scale_tensor,
                                               const std::optional<tensor::TensorPtr> &offset_tensor,
                                               const std::optional<tensor::TensorPtr> &bias_tensor,
                                               const std::optional<tensor::TensorPtr> &pertokenScaleOptional_tensor,
                                               const BoolImmPtr &transpose_x1, const BoolImmPtr &transpose_x2,
                                               const Int64ImmPtr &dtype) {
  auto output = InferOutput<1>(prim::kPrimQuantBatchMatmul, x1_tensor, x2_tensor, scale_tensor, offset_tensor,
                               bias_tensor, pertokenScaleOptional_tensor, transpose_x1, transpose_x2, dtype);

  auto transpose_x1_imm = GetValue<bool>(transpose_x1);
  auto transpose_x2_imm = GetValue<bool>(transpose_x2);

  TensorToDevice(x1_tensor, x2_tensor, scale_tensor, offset_tensor, bias_tensor, pertokenScaleOptional_tensor);
  TensorMalloc(output);

  DISPATCH_ACLNN(aclnnQuantMatmulV4, x1_tensor, x2_tensor, scale_tensor, offset_tensor, pertokenScaleOptional_tensor,
                 bias_tensor, transpose_x1_imm, transpose_x2_imm, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
