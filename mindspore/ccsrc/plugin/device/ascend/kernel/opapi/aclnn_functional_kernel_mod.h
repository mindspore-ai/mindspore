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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACLNN_FUNCTIONAL_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACLNN_FUNCTIONAL_KERNEL_MOD_H_
#include "transform/acl_ir/op_api_exec.h"

namespace mindspore {
namespace kernel {
using TensorParams = transform::TensorParams;
using aclOpExecutor = transform::aclOpExecutor;
using CallBackFunc = std::function<void()>;

class AclnnFunctionalKernelMod {
 public:
  AclnnFunctionalKernelMod() {}
  ~AclnnFunctionalKernelMod() = default;

  virtual void Init(const PrimitivePtr &prim, bool is_gradient_out);

  // TODO(wch) support ref and speacial format
  // Malloc device memory
  void CreateTensorAddress(const tensor::TensorPtr &tensor, const std::string &input_name,
                           bool is_gradient_out = false);
  device::DeviceAddressPtr CreateWorkspaceAddress(const size_t &workspace_size);

 protected:
  aclOpExecutor *executor_{nullptr};
  PrimitivePtr prim_;
  std::string kernel_name_;
  device::DeviceContext *device_context_;
  bool is_gradient_out_{false};
};

using AclnnFunctionalKernelModPtr = std::shared_ptr<AclnnFunctionalKernelMod>;
#define MS_ACLLNN_FUNCTIONAL_KERNEL_FACTORY_REG(NAME, DERIVE_CLASS) \
  MS_KERNEL_FACTORY_REG(AclnnFunctionalKernelMod, NAME, DERIVE_CLASS)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACLNN_FUNCTIONAL_KERNEL_MOD_H_
