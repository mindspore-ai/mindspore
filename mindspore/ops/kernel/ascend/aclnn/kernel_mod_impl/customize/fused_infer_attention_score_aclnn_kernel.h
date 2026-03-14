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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FUSED_INFER_ATTENTION_SCORE_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FUSED_INFER_ATTENTION_SCORE_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <utility>
#include <string>
#include "ops/base_operator.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"
#include "kernel/ascend/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
namespace fused_infer_attention_score {
class FusedInferAttentionScoreAscend : public AclnnKernelMod {
 public:
  FusedInferAttentionScoreAscend() : AclnnKernelMod(std::move("aclnnFusedInferAttentionScoreV2")) {}
  ~FusedInferAttentionScoreAscend() = default;
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  void SetScalarParam(const std::vector<KernelTensor *> &inputs);
  std::optional<std::vector<int64_t>> ConvertTensorToVector(const std::string &tensor_name,
                                                            KernelTensor *const tensor_ptr);
  DEFINE_GET_WORKSPACE_FOR_RESIZE()

  int64_t num_heads_ = 0;
  double scale_value_d_ = 0;
  int64_t pre_tokens_ = 0;
  int64_t next_tokens_ = 0;
  std::string input_layout_str_ = "";
  int64_t num_key_value_heads_ = 0;
  int64_t sparse_mode_ = 0;
  int64_t inner_precise_ = 0;
  int64_t block_size_ = 0;
  int64_t antiquant_mode_ = 0;
  bool softmax_lse_flag_ = false;
  int64_t key_antiquant_mode_ = 0;
  int64_t value_antiquant_mode_ = 0;
  std::vector<KernelTensor *> value_vector_;
  std::vector<KernelTensor *> key_vector_;
  std::pair<std::optional<std::vector<int64_t>>, bool> actual_q_lengths_vector_pair_;
  std::pair<std::optional<std::vector<int64_t>>, bool> actual_kv_lengths_vector_pair_;
  std::pair<std::optional<std::vector<int64_t>>, bool> actual_shared_prefix_lengths_vector_pair_;
};
}  // namespace fused_infer_attention_score
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FUSED_INFER_ATTENTION_SCORE_ACLNN_KERNEL_MOD_H_
