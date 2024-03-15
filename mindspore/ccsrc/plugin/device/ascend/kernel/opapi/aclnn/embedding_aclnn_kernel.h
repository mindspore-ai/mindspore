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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_EMBEDDING_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_EMBEDDING_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <utility>
#include <string>
#include "ops/base_operator.h"
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel_mod.h"
#include "transform/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {

class EmbeddingAscend : public AclnnKernelMod {
 public:
  EmbeddingAscend() : AclnnKernelMod(std::move("aclnnEmbedding")) {}
  ~EmbeddingAscend() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()

  void SetWorkspaceForRenorm(const KernelTensor *weight, const KernelTensor *input, double max_norm, double norm_type) {
    renorm_hash_id_ = transform::CalcOpApiHash(embedding_renorm_name_, weight, input, max_norm, norm_type);
    if (cache_hash_.count(renorm_hash_id_) == 0) {
      const bool use_huge_pages = false;
      auto return_value = GEN_EXECUTOR_CUST(embedding_renorm_name_, use_huge_pages, weight, input, max_norm, norm_type);
      UpdateRenormWorkspace(std::get<kWsSizeIndex>(return_value), false);
    } else {
      auto return_value =
        GEN_EXECUTOR_BOOST(embedding_renorm_name_, renorm_hash_id_, weight, input, max_norm, norm_type);
      UpdateRenormWorkspace(std::get<kWsSizeIndex>(return_value), true, std::get<kHashIdIndex>(return_value));
    }
  }

  inline void UpdateRenormWorkspace(uint64_t ws_size, bool boost, uint64_t new_hash_id = 0) {
    renorm_ws_size_ = ws_size;
    if (renorm_ws_size_ != 0) {
      workspace_size_list_.emplace_back(ws_size);
    }

    if (boost) {
      renorm_hash_id_ = new_hash_id;
    }
  }

  const std::string embedding_renorm_name_{"aclnnEmbeddingRenorm"};
  bool do_renorm_{false};
  bool renorm_ws_size_{0};
  uint64_t renorm_hash_id_{0};
  double max_norm_{0};
  double norm_type_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_EMBEDDING_ACLNN_KERNEL_MOD_H_
