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

#include "kernel/ascend/aclnn/pyboost_impl/customize/add_rmsnorm_quant_v2.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> AddRmsNormQuantAscendCustomize(
  const TensorPtr &x1_tensor, const TensorPtr &x2_tensor, const TensorPtr &gamma_tensor, const TensorPtr &scale_tensor,
  const TensorPtr &offset_tensor, const FP32ImmPtr &epsilon) {
  auto outputs = InferOutput<3>(prim::kPrimAddRmsNormQuantV2, x1_tensor, x2_tensor, gamma_tensor, scale_tensor,
                                offset_tensor, epsilon);
  // Convert ValuePtr to c++ scalar
  auto epsilon_imm = static_cast<double>(GetValue<float>(epsilon));

  TensorToDevice(x1_tensor, x2_tensor, gamma_tensor, scale_tensor, offset_tensor);
  TensorAllocate(outputs);

  const auto &y1 = std::get<0>(outputs);
  const auto &y2 = std::get<1>(outputs);
  const auto &x = std::get<2>(outputs);
  DISPATCH_ACLNN(aclnnAddRmsNormQuant, x1_tensor, x2_tensor, gamma_tensor, scale_tensor, nullptr, offset_tensor,
                 nullptr, nullptr, epsilon_imm, y1, y2, x);
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
