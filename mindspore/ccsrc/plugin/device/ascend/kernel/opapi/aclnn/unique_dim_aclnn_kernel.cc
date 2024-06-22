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
#include "plugin/device/ascend/kernel/opapi/aclnn/unique_dim_aclnn_kernel.h"
#include "ir/tensor.h"
#include "transform/acl_ir/op_api_convert.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {

void UniqueDimAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  auto sorted = transform::ConvertKernelTensor<bool>(inputs[kIndex1]);
  auto return_inverse = transform::ConvertKernelTensor<bool>(inputs[kIndex2]);
  auto dim = transform::ConvertKernelTensor<int64_t>(inputs[kIndex3]);
  GetWorkspaceForResize(inputs[kIndex0], sorted, return_inverse, dim, outputs[kIndex0], outputs[kIndex1],
                        outputs[kIndex2]);
}

bool UniqueDimAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  MS_LOG(DEBUG) << "Run UniqueDim start.";

  auto sorted = transform::ConvertKernelTensor<bool>(inputs[kIndex1]);
  auto return_inverse = transform::ConvertKernelTensor<bool>(inputs[kIndex2]);
  auto dim = transform::ConvertKernelTensor<int64_t>(inputs[kIndex3]);

  bool use_huge_pages = True;
  auto res = GEN_EXECUTOR_CUST(op_type_, use_huge_pages, inputs[kIndex0], sorted, return_inverse, dim, outputs[kIndex0],
                               outputs[kIndex1], outputs[kIndex2]);
  UpdateWorkspace(res);
  executor_ = std::get<kIndex1>(res);
  auto &all_acl_tensor = std::get<kIndex2>(res);
  RunOpSync(stream_ptr, workspace);
  MS_LOG(DEBUG) << "Run UniqueDim end.";

  // update output shape
  size_t output_size = 3;
  output_shapes_.resize(output_size);
  output_shapes_[kIndex0] = transform::UpdateOutputShape(all_acl_tensor.get<kIndex4>());
  output_shapes_[kIndex1] = transform::UpdateOutputShape(all_acl_tensor.get<kIndex5>());
  output_shapes_[kIndex2] = transform::UpdateOutputShape(all_acl_tensor.get<kIndex6>());
  return true;
}

void UniqueDimAscend::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  outputs[kIndex0]->SetShapeVector(output_shapes_[kIndex0]);
  outputs[kIndex1]->SetShapeVector(output_shapes_[kIndex1]);
  outputs[kIndex2]->SetShapeVector(output_shapes_[kIndex2]);
}

MS_ACLNN_KERNEL_FACTORY_REG(UniqueDim, UniqueDimAscend);
}  // namespace kernel
}  // namespace mindspore
