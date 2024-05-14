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
#include "plugin/device/ascend/kernel/opapi/aclnn/cross_ext_aclnn_kernel.h"
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
namespace mindspore {
namespace kernel {

int64_t CrossExtAscend::UpdateCrossDim(const ShapeVector &input_shape, const std::optional<int64_t> &dim_opt) {
  if (!dim_opt.has_value()) {
    int64_t dim_size_value = 3;
    for (size_t i = 0; i < input_shape.size(); i++) {
      if (input_shape[i] == dim_size_value) {
        return SizeToLong(i);
      }
    }
  }
  return dim_opt.value();
}

void CrossExtAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  auto dim_opt = inputs[kIndex2]->GetOptionalValueWithCheck<int64_t>();
  int64_t dim_imm = UpdateCrossDim(inputs[kIndex0]->GetShape()->GetShapeVector(), dim_opt);
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], dim_imm, outputs[kIndex0]);
}

bool CrossExtAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                            const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto input_shape = inputs[kIndex0]->GetShape()->GetShapeVector();
  auto dim_opt = inputs[kIndex2]->GetOptionalValueWithCheck<int64_t>();
  int64_t dim_imm = UpdateCrossDim(input_shape, dim_opt);
  ParseGenExecutor(GEN_EXECUTOR_BOOST(op_type_, hash_id_, inputs[kIndex0], inputs[kIndex1], dim_imm, outputs[kIndex0]));
  RunOp(stream_ptr, workspace);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(CrossExt, CrossExtAscend);
}  // namespace kernel
}  // namespace mindspore
