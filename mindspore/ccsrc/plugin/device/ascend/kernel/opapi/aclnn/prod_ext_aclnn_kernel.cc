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
#include "plugin/device/ascend/kernel/opapi/aclnn/prod_ext_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "transform/acl_ir/acl_helper.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
void ProdExtAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  const auto axis_opt = inputs[kIndex1]->GetOptionalValueWithCheck<std::vector<int64_t>>();
  if (axis_opt.has_value() && axis_opt.value().size() == 1) {
    axis_ = axis_opt.value()[0];
    keep_dims_ = transform::ConvertKernelTensor<bool>(inputs[kIndex2]);
    is_all_reduce_ = false;
  } else {
    is_all_reduce_ = true;
  }
  // Infer function has confirmed the actual dtype of output
  dtype_ = outputs[kIndex0]->dtype_id();

  if (is_all_reduce_) {
    op_type_ = std::move("aclnnProd");
    GetWorkspaceForResize(inputs[kIndex0], dtype_, outputs[kIndex0]);
  } else {
    op_type_ = std::move("aclnnProdDim");
    GetWorkspaceForResize(inputs[kIndex0], axis_, keep_dims_, dtype_, outputs[kIndex0]);
  }
}

bool ProdExtAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (is_all_reduce_) {
    RunOp(stream_ptr, workspace, inputs[kIndex0], dtype_, outputs[kIndex0]);
  } else {
    RunOp(stream_ptr, workspace, inputs[kIndex0], axis_, keep_dims_, dtype_, outputs[kIndex0]);
  }
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(ProdExt, ProdExtAscend);
}  // namespace kernel
}  // namespace mindspore
