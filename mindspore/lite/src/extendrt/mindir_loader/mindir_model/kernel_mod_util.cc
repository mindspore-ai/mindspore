/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <vector>
#include <memory>

#include "extendrt/mindir_loader/mindir_model/kernel_mod_util.h"

#include "kernel/kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/cpu_kernel_mod.h"

namespace mindspore::kernel {
std::shared_ptr<mindspore::kernel::InnerKernel> KernelModUtil::GetInnerKernel(
  const std::vector<mindspore::lite::Tensor *> &in_tensors, const std::vector<mindspore::lite::Tensor *> &out_tensors,
  const mindspore::lite::LiteGraph::Node *node, lite::InnerContext *context) {
  auto op_type = node->op_type_;
  std::shared_ptr<kernel::KernelMod> kernel_mod = nullptr;
  if (kernel::Factory<kernel::CpuKernelMod>::Instance().IsRegistered(op_type)) {
    kernel_mod = kernel::Factory<kernel::CpuKernelMod>::Instance().Create(op_type);
  }
  if (kernel_mod == nullptr) {
    return nullptr;
  }
  auto base_operator = std::reinterpret_pointer_cast<ops::BaseOperator>(node->base_operator_);
  return std::make_shared<kernel::InnerKernel>(kernel_mod, base_operator, in_tensors, out_tensors, context);
}
}  // namespace mindspore::kernel
