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

#include "kernel/ascend/aclnn/pyboost_impl/customize/cross.h"
#include "infer/ops_func_impl/cross.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

tensor::TensorPtr CrossAscendCustomize(const TensorPtr &input_tensor, const TensorPtr &other_tensor,
                                       const Int64ImmPtr &dim) {
  MS_LOG(DEBUG) << "Cross call start";
  auto output = InferOutput<1>(prim::kPrimCross, input_tensor, other_tensor, dim);
  auto dim_imm = GetValue<int64_t>(dim);
  const int64_t default_dim = -65530;
  if (dim_imm == default_dim) {
    const auto &input_shape = input_tensor->shape();
    const auto &other_shape = other_tensor->shape();
    dim_imm = SizeToLong(ops::CalCrossDimFromDefaultValue(input_shape, other_shape));
  }
  TensorToDevice(input_tensor, other_tensor);

  TensorMalloc(output);

  DISPATCH_ACLNN(aclnnLinalgCross, input_tensor, other_tensor, dim_imm, output);
  return output;
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
