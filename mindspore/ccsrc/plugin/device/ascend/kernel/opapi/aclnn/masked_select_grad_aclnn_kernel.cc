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
#include "plugin/device/ascend/kernel/opapi/aclnn/masked_select_grad_aclnn_kernel.h"
#include <algorithm>
#include <functional>
#include <memory>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/acl_helper.h"

namespace mindspore {
namespace kernel {

void InplaceZero::SetWorkspaceForInplaceZero(const KernelTensor *input, std::vector<size_t> *workspace_sizes) {
  zero_hash_id_ = transform::CalcOpApiHash(inplace_zero_str_, input);
  uint64_t ws_size;
  if (cache_hash_.count(zero_hash_id_) == 0) {
    const bool use_huge_pages = false;
    auto return_value = GEN_EXECUTOR_CUST(inplace_zero_str_, use_huge_pages, input);
    ws_size = std::get<kWsSizeIndex>(return_value);
  } else {
    auto return_value = GEN_EXECUTOR_BOOST(inplace_zero_str_, zero_hash_id_, input);
    ws_size = std::get<kWsSizeIndex>(return_value);
    zero_hash_id_ = std::get<kHashIdIndex>(return_value);
  }
  zero_ws_size_ = ws_size;
  if (zero_ws_size_ != 0) {
    workspace_sizes->emplace_back(ws_size);
  }
}

void InplaceZero::SetZeroKernelTensor(KernelTensor *kernel_tensor, void *device_ptr, void *stream_ptr) {
  void *ws_addr = zero_ws_size_ != 0 ? device_ptr : nullptr;
  transform::aclOpExecutor *executor;
  std::function<void()> release_func;
  std::tie(std::ignore, executor, release_func, std::ignore, std::ignore) =
    GEN_EXECUTOR_BOOST(inplace_zero_str_, zero_hash_id_, kernel_tensor);
  RUN_OP_API_ASYNC(inplace_zero_str_, ws_addr, zero_ws_size_, executor, stream_ptr, release_func);
}

bool InplaceZero::IsWorkSpaceSize() { return zero_ws_size_; }

void MaskedSelectGradAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs) {
  GetWorkspaceForResize(outputs[kIndex0], inputs[kIndex1], inputs[kIndex2]);
  inplace_zero_.SetWorkspaceForInplaceZero(inputs[kIndex0], &workspace_size_list_);
}

bool MaskedSelectGradAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &workspace,
                                            const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  void *zero_workspace_device_ptr = nullptr;
  if (inplace_zero_.IsWorkSpaceSize()) {
    zero_workspace_device_ptr = workspace.back()->device_ptr();
  }
  inplace_zero_.SetZeroKernelTensor(outputs[kIndex0], zero_workspace_device_ptr, stream_ptr);
  ParseGenExecutor(GEN_EXECUTOR_BOOST(op_type_, hash_id_, outputs[kIndex0], inputs[kIndex1], inputs[kIndex2]));
  RunOp(stream_ptr, workspace);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(MaskedSelectGrad, MaskedSelectGradAclnnKernelMod);
}  // namespace kernel
}  // namespace mindspore
