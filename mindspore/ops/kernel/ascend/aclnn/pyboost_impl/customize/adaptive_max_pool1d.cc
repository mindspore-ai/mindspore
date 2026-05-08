/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/aclnn/pyboost_impl/customize/adaptive_max_pool1d.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "include/pynative/utils/pyboost/auto_generate/adaptive_max_pool2d.h"
#include "infer/ops_func_impl/adaptive_pool_1d_check.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr> AdaptiveMaxPool1DAscendCustomize(
  const tensor::TensorPtr &input_x_tensor, const ValueTuplePtr &output_size) {
  auto input_shape = input_x_tensor->shape();
  auto output_size_val = ConvertValueTupleToVector<int64_t>(output_size);
  ops::CheckAdaptivePool1DInputs("AdaptiveMaxPool1D", input_shape, output_size_val.size());
  auto origin_shape_dim = SizeToLong(input_shape.size());

  // unsqueeze shape
  std::vector<int64_t> expand_input_shape;
  for (auto i = 0; i < origin_shape_dim - 1; i++) {
    expand_input_shape.emplace_back(input_shape[i]);
  }
  expand_input_shape.emplace_back(1);
  expand_input_shape.emplace_back(input_shape[origin_shape_dim - 1]);

  kernel::pyboost::RequireGradGuard require_grad_guard(false);
  auto input_x_imm = reshape(input_x_tensor, expand_input_shape);
  auto output_size_2d = std::make_shared<ValueTuple>(
    std::vector<ValuePtr>{std::make_shared<Int64Imm>(1), std::make_shared<Int64Imm>(output_size_val[0])});
  // call AdaptiveMaxPool2d
  auto adaptive_max_pool2d_op = CREATE_PYBOOST_OP(AdaptiveMaxPool2D, device::DeviceType::kAscend);
  tensor::TensorPtr output_adaptive_max_pool2d_tensor;
  tensor::TensorPtr output_adaptive_max_pool2d_indices;
  std::tie(output_adaptive_max_pool2d_tensor, output_adaptive_max_pool2d_indices) =
    adaptive_max_pool2d_op->Call(input_x_imm, output_size_2d);
  // squeeze shape
  auto shape_pool2d = output_adaptive_max_pool2d_tensor->shape();
  auto shape_pool2d_dim = shape_pool2d.size();
  std::vector<int64_t> squeeze_input_shape;
  if (shape_pool2d_dim <= kDim2) {
    MS_LOG(EXCEPTION) << "For AdaptiveMaxPool1DAscendCustomize, the value of shape_pool2d.size is invalid.";
  }
  constexpr size_t offset = 2;
  for (size_t i = 0; i < shape_pool2d_dim - offset; i++) {
    squeeze_input_shape.emplace_back(shape_pool2d[i]);
  }
  squeeze_input_shape.emplace_back(shape_pool2d[shape_pool2d_dim - 1]);
  auto output_tensor = reshape(output_adaptive_max_pool2d_tensor, squeeze_input_shape);
  auto output_indices = reshape(output_adaptive_max_pool2d_indices, squeeze_input_shape);
  return std::make_tuple(output_tensor, output_indices);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
