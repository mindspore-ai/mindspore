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

#include "kernel/ascend/aclnn/kernel_mod_impl/customize/nsa_compress_grad_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <string>
#include "kernel/ascend/acl_ir/op_api_convert.h"

namespace mindspore {
namespace kernel {
namespace nsa_compress_grad {

void NsaCompressGradAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  block_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex3]);
  stride_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex4]);
  const auto seq_vec = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex5]);
  seq_len_pair_ = std::make_pair(seq_vec, true);
  layout_ = "TND";
  seq_len_type_ = 0;
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], seq_len_pair_, block_, stride_,
                        seq_len_type_, layout_, outputs[kIndex0], outputs[kIndex1]);
}

bool NsaCompressGradAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspace,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  MS_LOG(DEBUG) << "Run aclnnNsaCompressGrad in kernel_mod_impl";
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], seq_len_pair_, block_, stride_,
        seq_len_type_, layout_, outputs[kIndex0], outputs[kIndex1]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(NsaCompressGrad, NsaCompressGradAscend);

}  // namespace nsa_compress_grad
}  // namespace kernel
}  // namespace mindspore
