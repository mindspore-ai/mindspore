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

#include "kernel/ascend/aclnn/pyboost_impl/customize/batch_norm_elemt.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr BatchNormElemtAscendCustomize(const TensorPtr &input_tensor,
                                                const std::optional<TensorPtr> &weight_tensor,
                                                const std::optional<TensorPtr> &bias_tensor,
                                                const std::optional<TensorPtr> &mean_tensor,
                                                const std::optional<TensorPtr> &invstd_tensor, const FP32ImmPtr &eps) {
  MS_LOG(DEBUG) << "BatchNormElemt call start";
  auto output = InferOutput<1>(prim::kPrimBatchNormElemt, input_tensor, weight_tensor, bias_tensor, mean_tensor,
                               invstd_tensor, eps);
  // ValueTuple to std::vector

  // Convert ValuePtr to c++ scalar
  // Convert ValuePtr to c++ scalar
  auto eps_imm = GetValue<float>(eps);

  TensorToDevice(input_tensor, weight_tensor, bias_tensor, mean_tensor, invstd_tensor);

  TensorAllocate(output);

  // Async
  double eps_imm_d = static_cast<double>(eps_imm);
  DISPATCH_ACLNN(aclnnBatchNormElemt, input_tensor, weight_tensor, bias_tensor, mean_tensor, invstd_tensor, eps_imm_d,
                 output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
