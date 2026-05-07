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

#include "kernel/ascend/aclnn/pyboost_impl/customize/add_layernorm_v2.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> AddLayerNormAscendCustomize(
  const TensorPtr &x1_tensor, const TensorPtr &x2_tensor, const TensorPtr &gamma_tensor, const TensorPtr &beta_tensor,
  const FP32ImmPtr &epsilon, const BoolImmPtr &additional_out) {
  auto outputs =
    InferOutput<4>(prim::kPrimAddLayerNormV2, x1_tensor, x2_tensor, gamma_tensor, beta_tensor, epsilon, additional_out);
  // Convert ValuePtr to c++ scalar
  auto epsilon_imm = static_cast<double>(GetValue<float>(epsilon));
  auto additional_out_imm = GetValue<bool>(additional_out);

  TensorToDevice(x1_tensor, x2_tensor, gamma_tensor, beta_tensor);
  TensorMalloc(outputs);

  const auto &y = std::get<0>(outputs);
  const auto &mean = std::get<1>(outputs);
  const auto &rstd = std::get<2>(outputs);
  const auto &x = std::get<3>(outputs);
  DISPATCH_ACLNN(aclnnAddLayerNorm, x1_tensor, x2_tensor, gamma_tensor, beta_tensor, nullptr, epsilon_imm,
                 additional_out_imm, y, mean, rstd, x);
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
