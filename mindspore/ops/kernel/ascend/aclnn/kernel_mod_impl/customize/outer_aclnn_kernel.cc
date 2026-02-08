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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/outer_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"

namespace mindspore {
namespace kernel {
namespace outer {

void OuterAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  ShapeVector input_realshape = inputs[kIndex0]->GetShapeVector();
  input_kernel_tensor_ = inputs[kIndex0]->CloneKernelTensor();
  int input_flatten_size = std::accumulate(input_realshape.begin(), input_realshape.end(), static_cast<int64_t>(1),
                                           std::multiplies<int64_t>());
  auto input_flatten_shape = ShapeVector{input_flatten_size, 1};
  input_kernel_tensor_->SetShapeVector(input_flatten_shape);

  size_t offset = 0;
  auto input_flatten_shape_ori = input_flatten_shape;
  auto input_flatten_shape_new = input_flatten_shape;
  std::vector<int64_t> strides_new = {1, 1};
  std::vector<int64_t> strides_ori = {1, 1};
  TensorStorageInfoPtr tensor_storage_info = std::make_shared<TensorStorageInfo>(
    input_flatten_shape_new, strides_new, offset, input_flatten_shape_ori, strides_ori, true);
  input_kernel_tensor_->set_tensor_storage_info(tensor_storage_info);
  GetWorkspaceForResize(input_kernel_tensor_.get(), inputs[kIndex1], outputs[kIndex0]);
}

bool OuterAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                         const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  input_kernel_tensor_->set_device_ptr(inputs[kIndex0]->device_ptr());
  RunOp(stream_ptr, workspace, input_kernel_tensor_.get(), inputs[kIndex1], outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(Outer, OuterAscend);
}  // namespace outer
}  // namespace kernel
}  // namespace mindspore
