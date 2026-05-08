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

#include "kernel/ascend/aclnn/pyboost_impl/customize/argmin_with_value.h"

#include <string>
#include <tuple>

#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr> ArgMinWithValueAscendCustomize(const TensorPtr &input_tensor,
                                                                                const Int64ImmPtr &axis,
                                                                                const BoolImmPtr &keep_dims) {
  auto outputs = InferOutput<2>(prim::kPrimArgMinWithValue, input_tensor, axis, keep_dims);
  // Convert ValuePtr to c++ scalar
  auto axis_imm = GetValue<int64_t>(axis);
  auto keep_dims_imm = GetValue<bool>(keep_dims);

  TensorToDevice(input_tensor);
  TensorMalloc(outputs);
  const auto &index = std::get<0>(outputs);
  const auto &values = std::get<1>(outputs);
  DISPATCH_ACLNN(aclnnMinDim, input_tensor, axis_imm, keep_dims_imm, values, index);
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
