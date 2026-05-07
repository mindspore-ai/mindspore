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

#include "kernel/ascend/aclnn/pyboost_impl/customize/gather_d_grad_v2.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr GatherDGradAscendCustomize(const TensorPtr &x, const Int64ImmPtr &dim, const TensorPtr &index,
                                             const TensorPtr &d_out) {
  MS_EXCEPTION_IF_NULL(dim);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(index);
  MS_EXCEPTION_IF_NULL(d_out);

  auto output = InferOutput<1>(prim::kPrimGatherDGradV2, x, dim, index, d_out);
  auto dim_value = dim->value();
  TensorToDevice(index, d_out);
  TensorMalloc(output);

  // Async
  DISPATCH_ACLNN(aclnnInplaceZero, output);
  DISPATCH_ACLNN(aclnnScatterAdd, output, dim_value, index, d_out, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
