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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_UPSAMPLE_BILINEAR2D_GRAD_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_UPSAMPLE_BILINEAR2D_GRAD_ACLNN_KERNEL_MOD_H_

#include <vector>
#include <utility>
#include "ops/base_operator.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"
#include "kernel/ascend/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
namespace upsample_bilinear2d_grad {

class UpsampleBilinear2DGradAscend final : public AclnnKernelMod {
 public:
  UpsampleBilinear2DGradAscend() : AclnnKernelMod(std::move("aclnnUpsampleBilinear2dBackwardV2")) {}
  ~UpsampleBilinear2DGradAscend() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()

  std::vector<int64_t> input_size_;
  std::vector<int64_t> output_size_;
  double scales_h_;
  double scales_w_;
  bool align_corners_;
};
}  // namespace upsample_bilinear2d_grad
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_UPSAMPLE_BILINEAR2D_GRAD_ACLNN_KERNEL_MOD_H_
