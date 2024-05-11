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
#include "plugin/device/ascend/kernel/opapi/aclnn/non_zero_aclnn_kernel.h"
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

void NonZeroAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  GetWorkspaceForResize(inputs[kIndex0], outputs[kIndex0]);
}

bool NonZeroAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto use_huge_pages = true;
  auto res = GEN_EXECUTOR_CUST(op_type_, use_huge_pages, inputs[kIndex0], outputs[kIndex0]);
  UpdateWorkspace(res);
  executor_ = std::get<1>(res);
  auto &all_tensor = std::get<2>(res);
  RunOpSync(stream_ptr, workspace);

  // Update output shape.
  outputs_shape_.resize(1);
  outputs_shape_[kIndex0] = transform::UpdateOutputShape(all_tensor.get<1>());
  return true;
}
void NonZeroAscend::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &,
                                             const std::vector<KernelTensor *> &outputs) {
  outputs[kIndex0]->SetShapeVector(outputs_shape_[kIndex0]);
}
MS_ACLNN_KERNEL_FACTORY_REG(NonZero, NonZeroAscend);
}  // namespace kernel
}  // namespace mindspore
