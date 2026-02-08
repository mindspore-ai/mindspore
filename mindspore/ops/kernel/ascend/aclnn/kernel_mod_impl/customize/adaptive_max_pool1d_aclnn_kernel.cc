/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/adaptive_max_pool1d_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace adaptive_max_pool1d {
void AdaptiveMaxPool1DAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  SetParaForPool2D(inputs, outputs);
  GetWorkspaceForResize(input_kernel_tensor_.get(), output_size_for_2d_, output_kernel_tensors_[kIndex0].get(),
                        output_kernel_tensors_[kIndex1].get());
}

bool AdaptiveMaxPool1DAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &workspace,
                                     const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  input_kernel_tensor_->set_device_ptr(inputs[kIndex0]->device_ptr());
  for (size_t i = 0; i < output_kernel_tensors_.size(); ++i) {
    output_kernel_tensors_[i]->set_device_ptr(outputs[i]->device_ptr());
  }
  RunOp(stream_ptr, workspace, input_kernel_tensor_.get(), output_size_for_2d_, output_kernel_tensors_[kIndex0].get(),
        output_kernel_tensors_[kIndex1].get());
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(AdaptiveMaxPool1D, AdaptiveMaxPool1DAscend);
}  // namespace adaptive_max_pool1d
}  // namespace kernel
}  // namespace mindspore
