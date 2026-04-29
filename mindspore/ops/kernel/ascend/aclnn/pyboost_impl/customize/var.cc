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

#include "kernel/ascend/aclnn/pyboost_impl/customize/var.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "primitive/auto_generate/gen_ops_primitive_v.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr VarAscendCustomize(const TensorPtr &input_tensor, const std::optional<ValueTuplePtr> &dim,
                                     const Int64ImmPtr &correction, const BoolImmPtr &keepdim) {
  auto output = InferOutput<1>(prim::kPrimVar, input_tensor, dim, correction, keepdim);

  std::vector<int64_t> dim_vector = ConvertValueTupleToVector<int64_t>(dim);
  auto correction_imm = GetValue<int64_t>(correction);
  auto keepdim_imm = GetValue<bool>(keepdim);

  TensorToDevice(input_tensor);
  TensorAllocate(output);
  DISPATCH_ACLNN(aclnnVarCorrection, input_tensor, dim_vector, correction_imm, keepdim_imm, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
