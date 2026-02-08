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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_NSA_SELECT_ATTENTION_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_NSA_SELECT_ATTENTION_ACLNN_KERNEL_MOD_H_

#include <vector>
#include <string>
#include <utility>
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"

namespace mindspore {
namespace kernel {
namespace nsa_select_attention {

class NsaSelectAttentionAscend : public AclnnKernelMod {
 public:
  NsaSelectAttentionAscend() : AclnnKernelMod(std::move("aclnnNsaSelectedAttention")) {}
  ~NsaSelectAttentionAscend() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()
  double scale_value_ = 0;
  int64_t head_num_ = 0;
  int64_t select_block_size_ = 0;
  int64_t select_block_count_ = 0;
  std::pair<std::vector<int64_t>, bool> actual_seq_qlen_vector_pair_;
  std::pair<std::vector<int64_t>, bool> actual_seq_kvlen_vector_pair_;
  std::string layout_str_ = "TND";
  int64_t sparse_mode_ = 2;
};
}  // namespace nsa_select_attention
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_NSA_SELECT_ATTENTION_ACLNN_KERNEL_MOD_H_
