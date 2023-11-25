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
#include "plugin/device/ascend/kernel/opapi/aclnn/argmin_with_value_aclnn_kernel.h"
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

namespace mindspore {
namespace kernel {

void ArgMinWithValueAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  auto axis = transform::ConvertKernelTensor<int64_t>(inputs[kIndex1]);
  auto keep_dims = transform::ConvertKernelTensor<bool>(inputs[kIndex2]);

  auto return_value = GEN_EXECUTOR(op_type_, inputs[kIndex0], axis, keep_dims, outputs[kIndex1], outputs[kIndex0]);
  UpdateWorkspace(return_value);
}

bool ArgMinWithValueAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspace,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto axis = transform::ConvertKernelTensor<int64_t>(inputs[kIndex1]);
  auto keep_dims = transform::ConvertKernelTensor<bool>(inputs[kIndex2]);

  ParseGenExecutor(GEN_EXECUTOR(op_type_, inputs[kIndex0], axis, keep_dims, outputs[kIndex1], outputs[kIndex0]));
  RunOp(stream_ptr, workspace);
  return true;
}
MS_ACLNN_KERNEL_FACTORY_REG(ArgMinWithValue, ArgMinWithValueAscend);
}  // namespace kernel
}  // namespace mindspore
