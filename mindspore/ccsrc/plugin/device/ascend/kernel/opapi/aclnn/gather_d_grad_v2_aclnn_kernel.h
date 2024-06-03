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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GATHER_D_GRAD_V2_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GATHER_D_GRAD_V2_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <string>
#include <utility>
#include "ops/base_operator.h"
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel_mod.h"
#include "transform/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {

class GatherDGradAscend : public AclnnKernelMod {
 public:
  GatherDGradAscend() : AclnnKernelMod(std::move("aclnnScatterAdd")) {}
  ~GatherDGradAscend() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()

  static constexpr size_t kWsSizeIndex = 0;
  static constexpr size_t kHashIdIndex = 3;

  void SetWorkspaceForInplaceZero(const KernelTensor *input) {
    zero_hash_id_ = transform::CalcOpApiHash(inplace_zero_str_, input);
    if (cache_hash_.count(zero_hash_id_) == 0) {
      const bool use_huge_pages = false;
      auto return_value = GEN_EXECUTOR_CUST(inplace_zero_str_, use_huge_pages, input);
      UpdateInplacemWorkspace(std::get<kWsSizeIndex>(return_value), false);
    } else {
      auto return_value = GEN_EXECUTOR_BOOST(inplace_zero_str_, zero_hash_id_, input);
      UpdateInplacemWorkspace(std::get<kWsSizeIndex>(return_value), true, std::get<kHashIdIndex>(return_value));
    }
  }

  inline void UpdateInplacemWorkspace(uint64_t ws_size, bool boost, uint64_t new_hash_id = 0) {
    zero_ws_size_ = ws_size;
    if (zero_ws_size_ != 0) {
      workspace_size_list_.emplace_back(ws_size);
    }

    if (boost) {
      zero_hash_id_ = new_hash_id;
    }
  }

  const std::string inplace_zero_str_{"aclnnInplaceZero"};
  bool zero_ws_size_{0};
  uint64_t zero_hash_id_{0};
  int64_t dim_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GATHER_D_GRAD_V2_ACLNN_KERNEL_MOD_H_
