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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACL_GETNEXT_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACL_GETNEXT_KERNEL_MOD_H_
#include <vector>
#include "plugin/device/ascend/kernel/acl/acl_kernel_mod.h"

namespace mindspore {
namespace kernel {
class GetNextAclKernelMod : public AclKernelMod {
 public:
  GetNextAclKernelMod() {}
  ~GetNextAclKernelMod() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool IsNeedUpdateOutputShapeAndSize() override { return true; }

  // Already update output shape and size in Resize, just skip UpdateOutputShapeAndSize impl.
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override {}
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACL_GETNEXT_KERNEL_MOD_H_
