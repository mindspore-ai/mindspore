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
#include "kernel/host/view/view_utils.h"
#include <vector>
#include <utility>
#include <memory>
#include <functional>
#include "ir/tensor_storage_info.h"
namespace mindspore {
namespace kernel {

ops::OldTensorInfoPtr GetOldTensorInfo(const KernelTensor *tensor) {
  if (tensor->tensor_storage_info() == nullptr) {
    auto old_strides = ops::GetOriStrides(tensor->GetShapeVector());
    return std::make_shared<ops::OldTensorInfo>(tensor->GetShapeVector(), old_strides, tensor->GetShapeVector(),
                                                old_strides, 0);
  } else {
    auto storage_info = tensor->tensor_storage_info();
    return std::make_shared<ops::OldTensorInfo>(storage_info->shape, storage_info->strides, storage_info->ori_shape,
                                                storage_info->ori_strides, storage_info->storage_offset);
  }
}

std::vector<int64_t> GetTensorStride(const KernelTensor *tensor) {
  const auto &storage_info = tensor->tensor_storage_info();
  if (storage_info != nullptr) {
    return storage_info->strides;
  }
  return ops::GetOriStrides(tensor->GetShapeVector());
}
}  // namespace kernel
}  // namespace mindspore
