/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/group_norm.h"
#include <functional>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/auto_generate/contiguous.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> GroupNormAscendCustomize(
  const TensorPtr &input_tensor, const Int64ImmPtr &num_groups, const std::optional<TensorPtr> &gamma_opt_tensor,
  const std::optional<TensorPtr> &beta_opt_tensor, const FP32ImmPtr &eps) {
  MS_LOG(DEBUG) << "Call start";
  // Convert ValuePtr to c++ scalar
  auto outputs = InferOutput<3>(prim::kPrimGroupNorm, input_tensor, num_groups, gamma_opt_tensor, beta_opt_tensor, eps);
  auto num_groups_imm = GetValue<int64_t>(num_groups);
  auto eps_imm = static_cast<double>(GetValue<float>(eps));

  const auto &shape = input_tensor->shape();
  const int64_t N = shape[0];
  const int64_t C = shape[1];
  const int64_t HxW = (shape.size() == kDim2) ? 1
                                              : std::accumulate(shape.begin() + 2, shape.end(), static_cast<int64_t>(1),
                                                                std::multiplies<int64_t>());
  TensorToDevice(input_tensor, gamma_opt_tensor, beta_opt_tensor);
  TensorMalloc(outputs);

  DISPATCH_ACLNN(aclnnGroupNorm, input_tensor, gamma_opt_tensor, beta_opt_tensor, N, C, HxW, num_groups_imm, eps_imm,
                 std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs));
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
