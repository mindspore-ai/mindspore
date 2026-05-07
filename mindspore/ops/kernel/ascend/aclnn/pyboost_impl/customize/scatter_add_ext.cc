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

#include "kernel/ascend/aclnn/pyboost_impl/customize/scatter_add_ext.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "ir/tensor.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr ScatterAddExtAscendCustomize(const TensorPtr &input_tensor, const Int64ImmPtr &dim,
                                               const TensorPtr &index_tensor, const TensorPtr &src_tensor) {
  auto output = InferOutput<1>(prim::kPrimScatterAddExt, input_tensor, dim, index_tensor, src_tensor);
  const auto dim_imm = GetValue<int64_t>(dim);
  TensorToDevice(input_tensor, index_tensor, src_tensor);
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnScatterAdd, input_tensor, dim_imm, index_tensor, src_tensor, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
