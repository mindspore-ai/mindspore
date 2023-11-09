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
#include "plugin/device/ascend/kernel/opapi/aclnn/abs_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/stream.h"
#include "runtime/device/kernel_runtime.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "transform/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
constexpr size_t kInputIndex = 0;
constexpr size_t kOutputIndex = 0;

void AbsAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  auto return_value = GEN_EXECUTOR(aclnnAbs, inputs[kInputIndex], outputs[kOutputIndex]);
  UpdateWorkspace(std::get<0>(return_value));
}

bool AbsAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  ParseGenExecutor(GEN_EXECUTOR(aclnnAbs, inputs[kInputIndex], outputs[kOutputIndex]));
  RunOp("aclnnAbs", stream_ptr, workspace);
  return true;
}
MS_ACLLNN_KERNEL_FACTORY_REG(Abs, AbsAclnnKernelMod);
}  // namespace kernel
}  // namespace mindspore
