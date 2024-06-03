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
#include "plugin/device/ascend/kernel/opapi/aclnn/concat_aclnn_kernel.h"
#include <utility>
#include "transform/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kConcatMinNum = 2;
std::pair<std::vector<KernelTensor *>, int64_t> GetConcatRealInputs(const std::vector<KernelTensor *> &inputs) {
  if (MS_UNLIKELY(inputs.size() < kConcatMinNum)) {
    MS_LOG(EXCEPTION) << "For 'Concat', inputs should be 2 at least, bug got " << inputs.size();
  }

  auto last_element = inputs.end() - 1;
  std::vector<KernelTensor *> tensors(inputs.begin(), last_element);
  if (inputs.size() == kConcatMinNum) {
    tensors.clear();
    tensors = transform::ConvertKernelTensor<std::vector<KernelTensor *>>(inputs[kIndex0]);
  }

  auto last_kernel_tensor = *last_element;
  MS_EXCEPTION_IF_NULL(last_kernel_tensor);
  auto axis = last_kernel_tensor->GetValueWithCheck<int64_t>();
  return std::make_pair(tensors, axis);
}
}  // namespace
void ConcatAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  std::tie(tensors_, axis_) = GetConcatRealInputs(inputs);
  GetWorkspaceForResize(tensors_, axis_, outputs[kIndex0]);
}

bool ConcatAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                          const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  std::tie(tensors_, axis_) = GetConcatRealInputs(inputs);
  RunOp(stream_ptr, workspace, tensors_, axis_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(Concat, ConcatAscend);
}  // namespace kernel
}  // namespace mindspore
