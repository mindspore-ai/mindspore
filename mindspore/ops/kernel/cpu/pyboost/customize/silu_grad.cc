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

#include "mindspore/ops/kernel/cpu/pyboost/customize/silu_grad.h"
#include "kernel/cpu/cpu_kernel.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/sigmoid.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/sigmoid_grad.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/mul.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/add_ext.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
OpPtr SiLUGradCPUCall(const device::DeviceContext *device_context, const TensorPtr &dout_tensor,
                      const TensorPtr &x_tensor) {
  MS_LOG(DEBUG) << "Call start";
  const auto &sigmoid = CREATE_PYBOOST_OP(Sigmoid, device::DeviceType::kCPU);
  const auto &mul_a = CREATE_PYBOOST_OP(Mul, device::DeviceType::kCPU);
  const auto &mul_b = CREATE_PYBOOST_OP(Mul, device::DeviceType::kCPU);
  const auto &sigmoid_grad = CREATE_PYBOOST_OP(SigmoidGrad, device::DeviceType::kCPU);
  const auto &add = CREATE_PYBOOST_OP(AddExt, device::DeviceType::kCPU);

  auto alpha = std::make_shared<FP32Imm>(1.0);

  const auto &sigmoid_tensor = sigmoid->Call(x_tensor);
  const auto &bc_dx = mul_a->Call(x_tensor, dout_tensor);
  const auto &bc_dy = mul_b->Call(sigmoid_tensor, dout_tensor);
  const auto &dx = sigmoid_grad->Call(sigmoid_tensor, bc_dx);
  add->Call(dx, bc_dy, alpha);
  MS_LOG(DEBUG) << "Launch end";
  return add;
}
}  // namespace

void SiLUGradCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &dout_tensor,
                          const TensorPtr &x_tensor) {
  auto device_context = op->device_context();
  const auto &output = SiLUGradCPUCall(device_context, dout_tensor, x_tensor);
  op->set_outputs(output->outputs());
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
