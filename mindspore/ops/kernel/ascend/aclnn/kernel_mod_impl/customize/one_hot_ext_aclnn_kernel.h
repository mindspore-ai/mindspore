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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ONE_HOT_EXT_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ONE_HOT_EXT_ACLNN_KERNEL_MOD_H_

#include <memory>
#include <vector>
#include <utility>
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"

namespace mindspore {
namespace kernel {
namespace one_hot_ext {

class OneHotExtAscend : public AclnnKernelMod {
 public:
  OneHotExtAscend() : AclnnKernelMod(std::move("aclnnOneHot")) {}

  ~OneHotExtAscend() = default;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool IsNeedUpdateOutputShapeAndSize() override { return need_update_output_shape_; }
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override;

 private:
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnMax, DoMax)
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnOneHot, DoOneHot)

  int64_t depth_;
  int64_t axis_;
  bool auto_depth_{false};
  bool need_update_output_shape_{false};
  KernelTensor *max_output_tensor_{nullptr};
  std::vector<ShapeVector> output_shape_{{0}};
  TypePtr origin_output_typeptr_;
};
}  // namespace one_hot_ext
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ONE_HOT_EXT_ACLNN_KERNEL_MOD_H_
