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
#include "kernel/pyboost/py_boost_utils.h"
#include "plugin/device/cpu/kernel/pyboost/auto_generate/sigmoid.h"
#include "plugin/device/cpu/kernel/pyboost/auto_generate/sigmoid_grad.h"
#include "plugin/device/cpu/kernel/pyboost/auto_generate/mul.h"
#include "plugin/device/cpu/kernel/pyboost/auto_generate/add.h"
#include "mindspore/core/ops/auto_generate/gen_ops_primitive.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
OpPtr SiLUGradCPUCall(const device::DeviceContext *device_context, const TensorPtr &dout_tensor,
                      const TensorPtr &x_tensor) {
  MS_LOG(DEBUG) << "Call start";
  const auto &sigmoid = CREATE_PYBOOST_OP(Sigmoid, device_context->device_context_key_.device_name_);
  sigmoid->set_primitive(prim::kPrimSigmoid);
  const auto &mul_a = CREATE_PYBOOST_OP(Mul, device_context->device_context_key_.device_name_);
  mul_a->set_primitive(prim::kPrimMul);
  const auto &mul_b = CREATE_PYBOOST_OP(Mul, device_context->device_context_key_.device_name_);
  mul_b->set_primitive(prim::kPrimMul);
  const auto &sigmoid_grad = CREATE_PYBOOST_OP(SigmoidGrad, device_context->device_context_key_.device_name_);
  sigmoid_grad->set_primitive(prim::kPrimSigmoidGrad);
  const auto &add = CREATE_PYBOOST_OP(Add, device_context->device_context_key_.device_name_);
  add->set_primitive(prim::kPrimAdd);

  const auto &sigmoid_tensor = sigmoid->Call(x_tensor);
  const auto &bc_dx = mul_a->Call(x_tensor, dout_tensor);
  const auto &bc_dy = mul_b->Call(sigmoid_tensor, dout_tensor);
  const auto &dx = sigmoid_grad->Call(sigmoid_tensor, bc_dx);
  add->Call(dx, bc_dy);
  MS_LOG(DEBUG) << "Launch end";
  return add;
}
}  // namespace

void SiLUGradCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &dout_tensor,
                          const TensorPtr &x_tensor) {
  auto device_context = op->device_context();
  const auto &output = SiLUGradCPUCall(device_context, dout_tensor, x_tensor);
  op->set_input_abs({dout_tensor->ToAbstract(), x_tensor->ToAbstract()});
  op->set_output_abs(output->output_abs());
  op->set_outputs(output->outputs());
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
