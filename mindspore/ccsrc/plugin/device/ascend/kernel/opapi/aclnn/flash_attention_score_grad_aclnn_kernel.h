/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FLASH_ATTENTION_SCORE_GRAD_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FLASH_ATTENTION_SCORE_GRAD_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <string>
#include "ops/base_operator.h"
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel_mod.h"
#include "transform/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
using TensorParams = transform::TensorParams;

class FAScoreGradAclnnKernelMod : public AclnnKernelMod {
 public:
  FAScoreGradAclnnKernelMod() : AclnnKernelMod("aclnnFlashAttentionScoreGrad") {}
  ~FAScoreGradAclnnKernelMod() = default;

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

 protected:
  auto FAGradGenerate(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
    auto scale_value = static_cast<double>(GetFAGradAttr<float>("scale_value"));
    auto keep_prob = static_cast<double>(GetFAGradAttr<float>("keep_prob"));
    auto pre_tokens = GetFAGradAttr<int64_t>("pre_tokens");
    auto next_tokens = GetFAGradAttr<int64_t>("next_tokens");
    auto head_num = GetFAGradAttr<int64_t>("head_num");
    auto input_layout = GetFAGradAttr<std::string>("input_layout");
    auto inner_precise = GetFAGradAttr<int64_t>("inner_precise");
    EmptyKernelTensor empty_tensor(outputs[0]->type_id(), outputs[0]->dtype_id());
    auto return_value =
      GEN_EXECUTOR(op_type_, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex7], nullptr, nullptr,
                   nullptr, inputs[kIndex3], inputs[kIndex5], inputs[kIndex6], nullptr, inputs[kIndex4], nullptr,
                   scale_value, keep_prob, pre_tokens, next_tokens, head_num, input_layout, inner_precise, nullptr,
                   outputs[kIndex0], outputs[kIndex1], outputs[kIndex2], empty_tensor.get());
    return return_value;
  }

  template <typename T>
  T GetFAGradAttr(const std::string &attr_name) {
    MS_EXCEPTION_IF_NULL(primitive());
    const auto &attr_list = primitive()->attrs();
    if (attr_list.at(attr_name) == nullptr) {
      MS_LOG(EXCEPTION) << "FlashAttentionGrad hasn't this attr:" << attr_name;
    }
    return GetValue<T>(attr_list.at(attr_name));
  }
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FLASH_ATTENTION_SCORE_GRAD_ACLNN_KERNEL_MOD_H_
