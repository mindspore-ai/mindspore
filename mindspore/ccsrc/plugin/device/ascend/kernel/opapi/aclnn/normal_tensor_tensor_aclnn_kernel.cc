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
#include "plugin/device/ascend/kernel/opapi/aclnn/normal_tensor_tensor_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {

void NormalTensorTensorAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &outputs) {
  constexpr int64_t seed_stub = 0;
  constexpr int64_t offset_stub = 0;
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], seed_stub, offset_stub, outputs[kIndex0]);
}

bool NormalTensorTensorAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &workspace,
                                      const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto seed = transform::ConvertKernelTensor<int64_t>(inputs[kIndex2]);
  auto offset = transform::ConvertKernelTensor<int64_t>(inputs[kIndex3]);
  ParseGenExecutor(
    GEN_EXECUTOR_BOOST(op_type_, hash_id_, inputs[kIndex0], inputs[kIndex1], seed, offset, outputs[kIndex0]));
  RunOp(stream_ptr, workspace);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(NormalTensorTensor, NormalTensorTensorAscend);
}  // namespace kernel
}  // namespace mindspore
