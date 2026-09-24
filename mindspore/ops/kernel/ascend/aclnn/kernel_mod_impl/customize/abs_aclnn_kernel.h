/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ABS_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ABS_ACLNN_KERNEL_MOD_H_

#include <memory>
#include <utility>
#include <vector>

#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"

namespace mindspore {
namespace kernel {
namespace abs {
class AbsAscend final : public AclnnKernelMod {
 public:
  AbsAscend() : AclnnKernelMod(std::move("aclnnAbs")) {}
  ~AbsAscend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  void SetRealImagView(const KernelTensor *input, const KernelTensorPtr &view, bool is_real) const;

  DEFINE_GET_WORKSPACE_FOR_RESIZE()
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnMul, SquareReal)
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnMul, SquareImag)
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnAdd, Add)
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnSqrt, Sqrt)

  bool use_complex_fallback_{false};
  KernelTensorPtr real_view_{nullptr};
  KernelTensorPtr imag_view_{nullptr};
  KernelTensorPtr real_square_{nullptr};
  KernelTensorPtr imag_square_{nullptr};
  KernelTensorPtr square_sum_{nullptr};
  ScalarPtr one_{nullptr};
};
}  // namespace abs
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ABS_ACLNN_KERNEL_MOD_H_
