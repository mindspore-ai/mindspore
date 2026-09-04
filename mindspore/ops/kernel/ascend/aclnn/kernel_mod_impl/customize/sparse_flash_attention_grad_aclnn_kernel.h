/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SPARSE_FLASH_ATTENTION_GRAD_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SPARSE_FLASH_ATTENTION_GRAD_ACLNN_KERNEL_MOD_H_

#include <vector>
#include <string>
#include <utility>
#include "ops/base_operator.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"
#include "kernel/ascend/acl_ir/acl_convert.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"

namespace mindspore {
namespace kernel {
namespace sparse_flash_attention_grad {

class SparseFlashAttentionGradAscend : public AclnnKernelMod {
 public:
  // Disable the MindSpore-side executor cache on purpose. The cache refreshes tensor addresses by
  // sequential executor indices (aclSetTensorAddr), which depends on the executor's internal tensor
  // layout and is fragile across CANN versions (e.g. the optional `sinks` input added by newer CANN).
  // Building an executor per call keeps this op robust regardless of the CANN tensor layout.
  SparseFlashAttentionGradAscend() : AclnnKernelMod(std::move("aclnnSparseFlashAttentionGrad")) { capacity_ = 0; }
  ~SparseFlashAttentionGradAscend() = default;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

 private:
  double scale_value_ = 1.0;
  int64_t sparse_block_size_ = 1;
  std::string layout_ = "BSND";
  int64_t sparse_mode_ = 3;
  int64_t pre_tokens_ = 9223372036854775807;
  int64_t next_tokens_ = 9223372036854775807;
  int64_t attention_mode_ = 0;
  bool deterministic_ = true;
  DEFINE_GET_WORKSPACE_FOR_RESIZE()
};

}  // namespace sparse_flash_attention_grad
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SPARSE_FLASH_ATTENTION_GRAD_ACLNN_KERNEL_MOD_H_
