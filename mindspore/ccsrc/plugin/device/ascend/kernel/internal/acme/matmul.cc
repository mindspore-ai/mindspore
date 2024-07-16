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

#include "plugin/device/ascend/kernel/internal/acme/matmul.h"

#include <memory>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
acme::AcmeOpPtr AcmeMatmul::CreateKernel(acme::InputsImmutableInfoList inputs, acme::OutputsImmutableInfoList outputs,
                                         const std::vector<KernelTensor *> &ms_inputs,
                                         const std::vector<KernelTensor *> &ms_outputs) {
  acme::MatmulParam param;
  param.transpose_a = ms_inputs[2]->GetValueWithCheck<bool>();
  param.transpose_b = ms_inputs[3]->GetValueWithCheck<bool>();
  const std::string op_name = "MatMul";
  return acme::CreateMatmulOp(inputs, outputs, param, op_name);
}
// MS_ACME_KERNEL_FACTORY_REG(MatMul, AcmeMatmul);
}  // namespace kernel
}  // namespace mindspore
