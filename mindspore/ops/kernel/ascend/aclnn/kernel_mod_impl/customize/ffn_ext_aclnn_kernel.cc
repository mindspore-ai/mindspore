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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/ffn_ext_aclnn_kernel.h"
#include <string>
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
namespace mindspore {
using mindspore::device::ascend::FFNActivationMode;
namespace kernel {
namespace ffn_ext {
void FFNExtAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  std::string activation_string = "fastgelu";
  auto activation_imm = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex14]);
  activation_string = FFNActivationMode::ConvertEnumToString(activation_imm);
  auto expertTokens = inputs[kIndex3];
  MS_EXCEPTION_IF_NULL(expertTokens);
  if (expertTokens->type_id() != kMetaTypeNone) {
    expertTokens_array = expertTokens->GetValueWithCheck<std::vector<int64_t>>();
  }
  innerPrecise_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex15]);
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], expertTokens_array, inputs[kIndex4],
                        inputs[kIndex5], inputs[kIndex6], inputs[kIndex7], inputs[kIndex8], inputs[kIndex9],
                        inputs[kIndex10], inputs[kIndex11], inputs[kIndex12], inputs[kIndex13], activation_string,
                        innerPrecise_, outputs[kIndex0]);
}

bool FFNExtAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                          const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  std::string activation_string = "fastgelu";
  auto activation_imm = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex14]);
  activation_string = FFNActivationMode::ConvertEnumToString(activation_imm);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], expertTokens_array, inputs[kIndex4],
        inputs[kIndex5], inputs[kIndex6], inputs[kIndex7], inputs[kIndex8], inputs[kIndex9], inputs[kIndex10],
        inputs[kIndex11], inputs[kIndex12], inputs[kIndex13], activation_string, innerPrecise_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(FFNExt, FFNExtAscend);
}  // namespace ffn_ext
}  // namespace kernel
}  // namespace mindspore
