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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_MASKED_SELECT_GRAD_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_MASKED_SELECT_GRAD_ACLNN_KERNEL_MOD_H_
#include <string>
#include <unordered_set>
#include <vector>
#include "ops/base_operator.h"
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel_mod.h"
#include "transform/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
class InplaceZero {
 public:
  void SetWorkspaceForInplaceZero(const KernelTensor *input, std::vector<size_t> *workspace_sizes);
  void SetZeroKernelTensor(KernelTensor *kernel_tensor, void *device_ptr, void *stream_ptr);
  bool IsWorkSpaceSize();

 private:
  const std::string inplace_zero_str_{"aclnnInplaceZero"};
  bool zero_ws_size_{0};
  uint64_t zero_hash_id_{0};
  std::unordered_set<uint64_t> cache_hash_;
  static constexpr size_t kWsSizeIndex = 0;
  static constexpr size_t kHashIdIndex = 3;
};

class MaskedSelectGradAclnnKernelMod : public AclnnKernelMod {
 public:
  MaskedSelectGradAclnnKernelMod() : AclnnKernelMod("aclnnInplaceMaskedScatter") {}
  ~MaskedSelectGradAclnnKernelMod() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  InplaceZero inplace_zero_;
  DEFINE_GET_WORKSPACE_FOR_RESIZE()
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_MASKED_SELECT_GRAD_ACLNN_KERNEL_MOD_H_
