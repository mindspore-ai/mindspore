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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_scatter_src_reduce.h"
#include <memory>
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "mindapi/base/types.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InplaceScatterSrcReduceAscendCustomize(const tensor::TensorPtr &input_tensor, const Int64ImmPtr &dim,
                                                         const tensor::TensorPtr &index_tensor,
                                                         const tensor::TensorPtr &src_tensor,
                                                         const Int64ImmPtr &reduce) {
  MS_LOG(DEBUG) << "Call InplaceScatterSrcReduce start";
  // No need to call infer
  TensorToDevice(input_tensor, index_tensor, src_tensor);

  auto dim_imm = GetValue<int64_t>(dim);
  auto reduce_imm = GetValue<int64_t>(reduce);
  // 0 means 'none' (replace) in aclnn, but should use scatter_ without reduce instead of using 'none'
  if ((reduce_imm != Reduce::ADD) && (reduce_imm != Reduce::MULTIPLY)) {
    MS_EXCEPTION(ValueError) << "For InplaceScatterSrcReduce, reduce must be either 'add' or 'multiply', but got: '"
                             << mindspore::device::ascend::ScatterReduceMode::ConvertEnumToString(reduce_imm) << "'.";
  }

  // Check Memory Partial Overlap
  mindspore::pyboost::CheckMemoryOverlap({input_tensor, index_tensor, src_tensor}, {input_tensor});
  DISPATCH_ACLNN(aclnnInplaceScatter, input_tensor, dim_imm, index_tensor, src_tensor, reduce_imm);
  return input_tensor;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
