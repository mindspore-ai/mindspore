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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_scatter_src.h"
#include <memory>
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InplaceScatterSrcAscendCustomize(const tensor::TensorPtr &input_tensor, const Int64ImmPtr &dim,
                                                   const tensor::TensorPtr &index_tensor,
                                                   const tensor::TensorPtr &src_tensor) {
  MS_LOG(DEBUG) << "Call InplaceScatterSrc start";
  // No need to call infer
  // forbid complex support: grad backward use aclnnGather which not supports complex
  if (!input_tensor->used_in_bprop_graph()) {
    TypeId input_type = input_tensor->Dtype()->type_id();
    if ((input_type == kNumberTypeComplex) || (input_type == kNumberTypeComplex64) ||
        (input_type == kNumberTypeComplex128)) {
      MS_EXCEPTION(RuntimeError)
        << "Complex Type is forbidden because the grad operation 'Gather' does not supports complex type";
    }
  }
  TensorToDevice(input_tensor, index_tensor, src_tensor);

  auto dim_imm = GetValue<int64_t>(dim);

  // Check Memory Partial Overlap
  mindspore::pyboost::CheckMemoryOverlap({input_tensor, index_tensor, src_tensor}, {input_tensor});
  const int64_t reduce = 0;  // reduce none
  DISPATCH_ACLNN(aclnnInplaceScatter, input_tensor, dim_imm, index_tensor, src_tensor, reduce);
  return input_tensor;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
