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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SOFTMAX_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SOFTMAX_ACLNN_KERNEL_MOD_H_

#include <vector>
#include <utility>
#include "ops/base_operator.h"
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel_mod.h"
#include "transform/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {

class SoftmaxAscend : public AclnnKernelMod {
 public:
  SoftmaxAscend() : AclnnKernelMod(std::move("aclnnSoftmax")) {}
  ~SoftmaxAscend() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  template <typename... Args>
  void GetWorkspaceForResize(const Args &... args) {
    hash_id_ = transform::CalcOpApiHash(args...);
    if (cache_hash_.count(hash_id_) == 0) {
      auto return_value = GEN_EXECUTOR_CUST(op_type_, args...);
      UpdateWorkspace(return_value);
    } else {
      auto return_value = GEN_EXECUTOR_BOOST(op_type_, hash_id_, args...);
      UpdateWorkspace(return_value);
    }
  }

  int64_t GetDimValue(KernelTensor *axis_ptr) const noexcept;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SOFTMAX_ACLNN_KERNEL_MOD_H_
