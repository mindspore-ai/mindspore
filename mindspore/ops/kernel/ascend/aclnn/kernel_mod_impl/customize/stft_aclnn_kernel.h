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
#ifndef MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_KERNEL_MOD_IMPL_CUSTOMIZE_STFT_ACLNN_KERNEL_H_
#define MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_KERNEL_MOD_IMPL_CUSTOMIZE_STFT_ACLNN_KERNEL_H_
#include <utility>
#include <vector>
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"
#include "kernel/ascend/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
namespace stft {
class STFTAscend : public AclnnKernelMod {
 public:
  STFTAscend() : AclnnKernelMod(std::move("aclStft")) {}
  ~STFTAscend() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnCast, CastInput)
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnCast, CastWindow)
  DEFINE_GET_WORKSPACE_FOR_OPS(aclStft, STFT)

  KernelTensor input_cast_;
  KernelTensor window_cast_;
  TypeId common_dtype_{kTypeUnknown};
  bool cast_input_{false};
  bool cast_window_{false};
  int64_t n_fft_{0};
  int64_t hop_length_{0};
  int64_t win_length_{0};
  bool normalized_{false};
  bool onesided_{false};
  bool return_complex_{false};
};
}  // namespace stft
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_KERNEL_MOD_IMPL_CUSTOMIZE_STFT_ACLNN_KERNEL_H_
