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

#include "plugin/device/cpu/kernel/pyboost/customize/silu_grad.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/cpu/kernel/pyboost/auto_generate/sigmoid.h"
#include "plugin/device/cpu/kernel/pyboost/auto_generate/sigmoid_grad.h"
#include "plugin/device/cpu/kernel/pyboost/auto_generate/mul.h"
#include "plugin/device/cpu/kernel/pyboost/auto_generate/add_ext.h"
#include "mindspore/core/ops/auto_generate/gen_ops_primitive.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
OpPtr SiLUGradCPUCall(const device::DeviceContext *device_context, const BaseTensorPtr &dout_tensor,
                      const BaseTensorPtr &x_tensor) {
  MS_LOG(DEBUG) << "Call start";
  const auto &sigmoid = CREATE_PYBOOST_OP(Sigmoid, device_context->device_context_key_.device_name_);
  const auto &mul_a = CREATE_PYBOOST_OP(Mul, device_context->device_context_key_.device_name_);
  const auto &mul_b = CREATE_PYBOOST_OP(Mul, device_context->device_context_key_.device_name_);
  const auto &sigmoid_grad = CREATE_PYBOOST_OP(SigmoidGrad, device_context->device_context_key_.device_name_);
  const auto &add = CREATE_PYBOOST_OP(AddExt, device_context->device_context_key_.device_name_);

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

void SiLUGradCPUCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &dout_tensor,
                          const BaseTensorPtr &x_tensor) {
  auto device_context = op->device_context();
  const auto &output = SiLUGradCPUCall(device_context, dout_tensor, x_tensor);
  op->set_outputs(output->outputs());
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
