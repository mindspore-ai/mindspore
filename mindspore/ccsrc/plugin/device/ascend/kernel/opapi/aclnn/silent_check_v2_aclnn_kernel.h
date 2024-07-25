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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SILENT_CHECK_V2_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SILENT_CHECK_V2_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <utility>
#include <string>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel_mod.h"
#include "transform/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {

class SilentCheckV2Ascend : public AclnnKernelMod {
 public:
  SilentCheckV2Ascend() : AclnnKernelMod(std::move("aclnnSilentCheck")) {}
  ~SilentCheckV2Ascend() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()

  void SetWorkspaceForInplaceCopy(const KernelTensor *output, const KernelTensor *input, size_t order) {
    copy_hash_ids_[order] = transform::CalcOpApiHash(inplace_copy_str_, input);
    auto copy_hash_id = copy_hash_ids_[order];
    if (cache_hash_.count(copy_hash_id) == 0) {
      const bool use_huge_pages = false;
      auto return_value = GEN_EXECUTOR_CUST(inplace_copy_str_, use_huge_pages, output, input);
      UpdateInplacemWorkspace(std::get<kWsSizeIndex>(return_value), false, order, order);
    } else {
      auto return_value = GEN_EXECUTOR_BOOST(inplace_copy_str_, copy_hash_id, output, input);
      UpdateInplacemWorkspace(std::get<kWsSizeIndex>(return_value), true, std::get<kHashIdIndex>(return_value), order);
    }
  }

  inline void UpdateInplacemWorkspace(uint64_t ws_size, bool boost, uint64_t new_hash_id = 0, size_t order = 0) {
    copy_ws_sizes_[order] = ws_size;
    if (ws_size != 0) {
      workspace_size_list_.emplace_back(ws_size);
    }

    if (boost) {
      copy_hash_ids_[order] = new_hash_id;
    }
  }

  const std::string inplace_copy_str_{"aclnnInplaceCopy"};
  std::vector<uint64_t> copy_ws_sizes_{0, 0, 0};
  std::vector<uint64_t> copy_hash_ids_{0, 0, 0};

  int64_t c_min_steps_{7};
  pyfloat c_thresh_l1_{1000000.};
  pyfloat c_coeff_l1_{100000.};
  pyfloat c_thresh_l2_{10000.};
  pyfloat c_coeff_l2_{5000.};
  int64_t npu_asd_detect_{1};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SILENT_CHECK_V2_ACLNN_KERNEL_MOD_H_
