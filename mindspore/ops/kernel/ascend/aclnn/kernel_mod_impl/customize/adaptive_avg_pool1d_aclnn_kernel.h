/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ADAPTIVE_AVG_POOL1D_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ADAPTIVE_AVG_POOL1D_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <utility>
#include <memory>
#include <string>
#include "ops/base_operator.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"
#include "kernel/ascend/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
namespace adaptive_avg_pool1d {
class AdaptivePool1DAscend : public AclnnKernelMod {
 public:
  explicit AdaptivePool1DAscend(std::string &&op_type) : AclnnKernelMod(std::move(op_type)) {}
  ~AdaptivePool1DAscend() = default;
  void SetParaForPool2D(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

 protected:
  std::vector<int64_t> output_size_for_2d_;
  ShapeVector out_shape_ori;
  std::shared_ptr<KernelTensor> input_kernel_tensor_;
  std::vector<std::shared_ptr<KernelTensor>> output_kernel_tensors_;
  DEFINE_GET_WORKSPACE_FOR_RESIZE()
};

class AdaptiveAvgPool1DAscend : public AdaptivePool1DAscend {
 public:
  AdaptiveAvgPool1DAscend() : AdaptivePool1DAscend(std::move("aclnnAdaptiveAvgPool2d")) {}
  ~AdaptiveAvgPool1DAscend() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
};

}  // namespace adaptive_avg_pool1d
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ADAPTIVE_AVG_POOL1D_ACLNN_KERNEL_MOD_H_
