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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/gelu_grad_ext_aclnn_kernel.h"

#include <algorithm>
#include <cstdint>
#include <vector>
#include <map>
#include <memory>
#include <unordered_map>
#include <functional>

#include "include/op_enum.h"
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace gelu_grad_ext {
namespace {
using ops::Approximate;
const std::unordered_map<Approximate, std::string> ApproximateModeMap{{Approximate::NONE, "none"},
                                                                      {Approximate::TANH, "tanh"}};
std::string GetApproximateMode(int64_t approximate) {
  auto approximate_enum = static_cast<Approximate>(approximate);
  auto it = ApproximateModeMap.find(approximate_enum);
  if (it == ApproximateModeMap.end()) {
    MS_EXCEPTION(ValueError) << "The value of approximate should be 0 or 1, but got " << approximate;
  }
  return it->second;
}
}  // namespace

void GeluGradExtAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  approximate_ = GetApproximateMode(inputs[kIndex2]->GetValueWithCheck<int64_t>());
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], approximate_, outputs[kIndex0]);
}

bool GeluGradExtAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], approximate_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(GeluGradExt, GeluGradExtAscend);
}  // namespace gelu_grad_ext
}  // namespace kernel
}  // namespace mindspore
