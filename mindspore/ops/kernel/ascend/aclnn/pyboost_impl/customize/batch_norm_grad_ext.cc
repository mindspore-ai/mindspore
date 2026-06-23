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

#include "kernel/ascend/aclnn/pyboost_impl/customize/batch_norm_grad_ext.h"
#include <algorithm>
#include <functional>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> BatchNormGradExtAscendCustomize(
  const TensorPtr &dout_tensor, const TensorPtr &input_tensor, const std::optional<TensorPtr> &weight_tensor,
  const std::optional<TensorPtr> &running_mean_tensor, const std::optional<TensorPtr> &runnning_var_tensor,
  const std::optional<TensorPtr> &saved_mean_tensor, const std::optional<TensorPtr> &saved_rstd_tensor,
  const BoolImmPtr &training, const FP32ImmPtr &eps, const ValueTuplePtr &output_mask) {
  constexpr size_t kOutputDx = 0;
  constexpr size_t kOutputDweight = 1;
  constexpr size_t kOutputDbias = 2;
  MS_LOG(DEBUG) << "Call aclnnBatchNormBackward start";
  // Convert ValuePtr to c++ scalar
  auto outputs =
    InferOutput<3>(prim::kPrimBatchNormGradExt, dout_tensor, input_tensor, weight_tensor, running_mean_tensor,
                   runnning_var_tensor, saved_mean_tensor, saved_rstd_tensor, training, eps, output_mask);
  auto training_imm = GetValue<bool>(training);
  auto eps_imm = static_cast<double>(GetValue<float>(eps));
  std::vector<int64_t> output_mask_vector = ConvertValueTupleToVector<int64_t>(output_mask);
  std::vector<uint8_t> output_mask_u8_vec;
  std::transform(output_mask_vector.begin(), output_mask_vector.end(), std::back_inserter(output_mask_u8_vec),
                 [](const int64_t &value) { return static_cast<uint8_t>(value); });

  TensorToDevice(dout_tensor, input_tensor, weight_tensor, running_mean_tensor, runnning_var_tensor, saved_mean_tensor,
                 saved_rstd_tensor);
  TensorMalloc(outputs);

  // Async
  DISPATCH_ACLNN(aclnnBatchNormBackward, dout_tensor, input_tensor, weight_tensor, running_mean_tensor,
                 runnning_var_tensor, saved_mean_tensor, saved_rstd_tensor, training_imm, eps_imm, output_mask_u8_vec,
                 std::get<kOutputDx>(outputs), std::get<kOutputDweight>(outputs), std::get<kOutputDbias>(outputs));
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
