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
#include <memory>
#include "kernel/ascend/aclop/kernel_mod_impl/acl_kernel_mod.h"
#include "include/runtime/hardware_abstract/data_queue/blocking_queue.h"

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

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
};

std::shared_ptr<device::BlockingQueue> GetTdtWingManQueue(const PrimitivePtr &prim);
std::shared_ptr<device::BlockingQueue> GetTdtWingManQueue(const std::shared_ptr<AnfNode> &node);
void CloseTdtWingManQueue(const PrimitivePtr &prim);
void CloseTdtWingManQueue(const std::shared_ptr<AnfNode> &node);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACL_GETNEXT_KERNEL_MOD_H_
