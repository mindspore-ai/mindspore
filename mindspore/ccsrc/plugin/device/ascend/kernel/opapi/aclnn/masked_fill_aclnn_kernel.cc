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
#include "plugin/device/ascend/kernel/opapi/aclnn/masked_fill_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/acl_helper.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/symbol_utils.h"

namespace mindspore {
namespace kernel {

void MaskedFillAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  GetWorkspaceForResize(outputs[kIndex0], inputs[kIndex1], inputs[kIndex2]);
  SetWorkspaceForInplaceCopy(outputs[kIndex0], inputs[kIndex0]);
}

bool MaskedFillAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  void *ws_addr = copy_ws_size_ != 0 ? workspace.back()->device_ptr() : nullptr;
  transform::aclOpExecutor *executor;
  std::function<void()> release_func;
  std::tie(std::ignore, executor, release_func, std::ignore, std::ignore) =
    GEN_EXECUTOR_BOOST(inplace_copy_str_, copy_hash_id_, outputs[kIndex0], inputs[kIndex0]);
  RUN_OP_API_ASYNC(inplace_copy_str_, ws_addr, copy_ws_size_, executor, stream_ptr, release_func);

  RunOp(stream_ptr, workspace, outputs[kIndex0], inputs[kIndex1], inputs[kIndex2]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(MaskedFill, MaskedFillAscend);
}  // namespace kernel
}  // namespace mindspore
