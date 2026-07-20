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

#include "kernel/ascend/aclnn/kernel_mod_impl/customize/upsample_nearest2d_aclnn_kernel.h"

#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>

#include "ir/dtype/type.h"
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace upsample_nearest2d {
void UpsampleNearest2DAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  auto input_dtype_id = inputs[0]->dtype_id();
  if (input_dtype_id == TypeId::kNumberTypeFloat64 || input_dtype_id == TypeId::kNumberTypeDouble) {
    MS_EXCEPTION(ValueError) << "For " << primitive_->name()
                             << ", input's type should not be float64, which is not supported.";
  }
  auto output_shape = outputs[kIndex0]->GetShapeVector();
  output_size_.clear();
  output_size_.assign(output_shape.begin() + kIndex2, output_shape.end());
  GetWorkspaceForResize(inputs[0], output_size_, outputs[0]);
}

bool UpsampleNearest2DAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &workspace,
                                     const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[0], output_size_, outputs[0]);

  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(UpsampleNearest2D, UpsampleNearest2DAscend);
}  // namespace upsample_nearest2d
}  // namespace kernel
}  // namespace mindspore
