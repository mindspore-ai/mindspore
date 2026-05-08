/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/reduce_all.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

tensor::TensorPtr ReduceAllAscendCustomize(const tensor::TensorPtr &input_tensor,
                                           const std::optional<ValueTuplePtr> &axis, const BoolImmPtr &keep_dims) {
  auto output = InferOutput<1>(prim::kPrimReduceAll, input_tensor, axis, keep_dims);

  std::vector<int64_t> axis_vector{};
  // If axis is not None, Convert ValueTuple to std::vector
  if (axis.has_value()) {
    axis_vector = ConvertValueTupleToVector<int64_t>(axis.value());
  }

  // Convert ValuePtr to c++ scalar
  bool keep_dims_imm = GetValue<bool>(keep_dims);

  TensorToDevice(input_tensor);
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnAll, input_tensor, axis_vector, keep_dims_imm, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
