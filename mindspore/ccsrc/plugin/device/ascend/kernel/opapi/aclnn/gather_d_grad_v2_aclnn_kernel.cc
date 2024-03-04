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
#include "plugin/device/ascend/kernel/opapi/aclnn/gather_d_grad_v2_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {

void GatherDGradAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  auto dim = transform::ConvertKernelTensor<int64_t>(inputs[kIndex1]);
  GetWorkspaceForResize(inputs[kIndex0], dim, inputs[kIndex2], inputs[kIndex3], outputs[kIndex0]);
}

bool GatherDGradAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto dim = transform::ConvertKernelTensor<int64_t>(inputs[kIndex1]);

  const std::string inplace_zero_str("aclnnInplaceZero");
  auto [ws_size, executor_handle, release_function] = GEN_EXECUTOR(inplace_zero_str, outputs[kIndex0]);
  MS_EXCEPTION_IF_CHECK_FAIL(ws_size == 0, "the workspace of aclnnInplaceZero is not zero!");
  RUN_OP_API_ASYNC(inplace_zero_str, nullptr, 0, executor_handle, stream_ptr, release_function);
  ParseGenExecutor(
    GEN_EXECUTOR_BOOST(op_type_, hash_id_, outputs[kIndex0], dim, inputs[kIndex2], inputs[kIndex3], outputs[kIndex0]));
  RunOp(stream_ptr, workspace);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(GatherDGradV2, GatherDGradAscend);
}  // namespace kernel
}  // namespace mindspore
