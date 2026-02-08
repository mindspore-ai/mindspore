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
#ifndef MINDSPORE_OPS_KERNEL_HOST_VIEW_UTILS_H_
#define MINDSPORE_OPS_KERNEL_HOST_VIEW_UTILS_H_
#include <vector>
#include "kernel/host/visible.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "ir/tensor_storage_info.h"
#include "view/view_strides_calculator.h"

namespace mindspore {
namespace kernel {
OPS_HOST_EXPORT ops::OldTensorInfoPtr GetOldTensorInfo(const KernelTensor *tensor);
std::vector<int64_t> GetTensorStride(const KernelTensor *tensor);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_OPS_KERNEL_HOST_VIEW_UTILS_H_
