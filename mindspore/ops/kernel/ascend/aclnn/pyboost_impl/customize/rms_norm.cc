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

#include "kernel/ascend/aclnn/pyboost_impl/customize/rms_norm.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr> RmsNormAscendCustomize(const TensorPtr &x_tensor,
                                                                        const TensorPtr &gamma_tensor,
                                                                        const FP32ImmPtr &epsilon) {
  auto outputs = InferOutput<2>(prim::kPrimRmsNorm, x_tensor, gamma_tensor, epsilon);
  auto epsilon_imm = static_cast<double>(GetValue<float>(epsilon));

  TensorToDevice(x_tensor, gamma_tensor);
  TensorAllocate(outputs);
  auto [o0, o1] = outputs;
  DISPATCH_ACLNN(aclnnRmsNorm, x_tensor, gamma_tensor, epsilon_imm, o0, o1);
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
