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

#include "kernel/ascend/aclnn/pyboost_impl/customize/batch_norm_gather_stats_with_counts.h"
#include <tuple>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr> BatchNormGatherStatsWithCountsAscendCustomize(
  const TensorPtr &input_tensor, const TensorPtr &mean_tensor, const TensorPtr &invstd_tensor,
  const std::optional<TensorPtr> &running_mean_tensor, const std::optional<TensorPtr> &running_var_tensor,
  const FP32ImmPtr &momentum, const FP32ImmPtr &eps, const std::optional<TensorPtr> &counts_tensor) {
  MS_LOG(DEBUG) << "BatchNormGatherStatsWithCounts call start";
  auto outputs = InferOutput<2>(prim::kPrimBatchNormGatherStatsWithCounts, input_tensor, mean_tensor, invstd_tensor,
                                running_mean_tensor, running_var_tensor, momentum, eps, counts_tensor);
  auto [output0, output1] = outputs;
  // ValueTuple to std::vector

  // Convert ValuePtr to c++ scalar
  // Convert ValuePtr to c++ scalar
  auto momentum_imm = GetValue<float>(momentum);
  auto eps_imm = GetValue<float>(eps);

  TensorToDevice(input_tensor, mean_tensor, invstd_tensor, running_mean_tensor, running_var_tensor, counts_tensor);

  TensorMalloc(outputs);

  // Async
  double momentum_imm_d = static_cast<double>(momentum_imm);
  double eps_imm_d = static_cast<double>(eps_imm);
  DISPATCH_ACLNN(aclnnBatchNormGatherStatsWithCounts, input_tensor, mean_tensor, invstd_tensor, running_mean_tensor,
                 running_var_tensor, momentum_imm_d, eps_imm_d, counts_tensor, output0, output1);
  return std::make_tuple(output0, output1);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
