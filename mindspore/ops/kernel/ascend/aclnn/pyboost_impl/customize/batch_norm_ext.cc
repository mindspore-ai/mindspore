/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/batch_norm_ext.h"
#include <functional>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> BatchNormExtAscendCustomize(
  const TensorPtr &input_tensor, const std::optional<TensorPtr> &weight_tensor,
  const std::optional<TensorPtr> &bias_tensor, const std::optional<TensorPtr> &mean_tensor,
  const std::optional<TensorPtr> &variance_tensor, const BoolImmPtr &training, const FP32ImmPtr &momentum,
  const FP32ImmPtr &epsilon) {
  MS_LOG(DEBUG) << "Call aclnnBatchNorm start";
  // Convert ValuePtr to c++ scalar
  auto outputs = InferOutput<3>(prim::kPrimBatchNormExt, input_tensor, weight_tensor, bias_tensor, mean_tensor,
                                variance_tensor, training, momentum, epsilon);
  auto training_imm = GetValue<bool>(training);
  auto momentum_imm = static_cast<double>(GetValue<float>(momentum));
  auto eps_imm = static_cast<double>(GetValue<float>(epsilon));

  TensorToDevice(input_tensor, weight_tensor, bias_tensor, mean_tensor, variance_tensor);
  TensorAllocate(outputs);
  DISPATCH_ACLNN(aclnnBatchNorm, input_tensor, weight_tensor, bias_tensor, mean_tensor, variance_tensor, training_imm,
                 momentum_imm, eps_imm, std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs));
  MS_LOG(DEBUG) << "Launch aclnnBatchNorm end";
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
