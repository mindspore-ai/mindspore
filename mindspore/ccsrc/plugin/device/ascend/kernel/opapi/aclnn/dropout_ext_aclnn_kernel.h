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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_OPAPI_ACLNN_DROPOUT_EXT_ACLNN_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_OPAPI_ACLNN_DROPOUT_EXT_ACLNN_KERNEL_H_
#include <vector>
#include <string>
#include <utility>
#include "ops/base_operator.h"
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel_mod.h"
#include "transform/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {

class DropoutExtAscend : public AclnnKernelMod {
 public:
  DropoutExtAscend() : AclnnKernelMod(std::move("aclnnDropout")) {}
  ~DropoutExtAscend() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnDropoutGenMaskV2, GenMask)
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnDropoutDoMask, DoMask)
  const std::string dropout_gen_mask_{"aclnnDropoutGenMaskV2"};
  const std::string dropout_do_mask_{"aclnnDropoutDoMask"};
  uint64_t gen_mask_hash_id_{0};
  uint64_t do_mask_hash_id_{0};
  double p_value_;
  int64_t seed_value_;
  int64_t offset_value_;
  TypeId dtype_value_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_OPAPI_ACLNN_DROPOUT_EXT_ACLNN_KERNEL_H_
