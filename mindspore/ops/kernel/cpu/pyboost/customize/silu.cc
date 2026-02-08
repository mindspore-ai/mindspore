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

#include "mindspore/ops/kernel/cpu/pyboost/customize/silu.h"
#include "kernel/cpu/cpu_kernel.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/sigmoid.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/mul.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
OpPtr SiLUCPUCall(const device::DeviceContext *device_context, const TensorPtr &x_tensor) {
  MS_LOG(DEBUG) << "Call start";
  const auto &sigmoid = CREATE_PYBOOST_OP(Sigmoid, device::DeviceType::kCPU);
  const auto &mul = CREATE_PYBOOST_OP(Mul, device::DeviceType::kCPU);
  const auto &sigmoid_tensor = sigmoid->Call(x_tensor);
  mul->Call(x_tensor, sigmoid_tensor);
  MS_LOG(DEBUG) << "Launch end";
  return mul;
}
}  // namespace

void SiLUCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor) {
  auto device_context = op->device_context();
  const auto &output = SiLUCPUCall(device_context, x_tensor);
  op->set_outputs(output->outputs());
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
