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

#include "kernel/ascend/aclnn/pyboost_impl/customize/adamw.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> AdamWAscendCustomize(
  const TensorPtr &var, const TensorPtr &m, const TensorPtr &v, const TensorPtr &max_v, const TensorPtr &grad,
  const TensorPtr &step, const FP32ImmPtr &lr, const FP32ImmPtr &beta1, const FP32ImmPtr &beta2,
  const FP32ImmPtr &decay, const FP32ImmPtr &epsilon, const BoolImmPtr &amsgrad, const BoolImmPtr &maximize) {
  const auto lr_imm = GetValue<float>(lr);
  const auto beta1_imm = GetValue<float>(beta1);
  const auto beta2_imm = GetValue<float>(beta2);
  const auto decay_imm = GetValue<float>(decay);
  const auto epsilon_imm = GetValue<float>(epsilon);
  const auto amsgrad_imm = GetValue<bool>(amsgrad);
  const auto maximize_imm = GetValue<bool>(maximize);

  if (amsgrad_imm) {
    TensorToDevice(var, m, v, max_v, grad, step);
  } else {
    TensorToDevice(var, m, v, grad, step);
  }

  if (amsgrad_imm) {
    DISPATCH_ACLNN(aclnnApplyAdamWV2, var, m, v, max_v, grad, step, lr_imm, beta1_imm, beta2_imm, decay_imm,
                   epsilon_imm, amsgrad_imm, maximize_imm);
  } else {
    DISPATCH_ACLNN(aclnnApplyAdamWV2, var, m, v, nullptr, grad, step, lr_imm, beta1_imm, beta2_imm, decay_imm,
                   epsilon_imm, amsgrad_imm, maximize_imm);
  }
  return std::make_tuple(var, m, v);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
