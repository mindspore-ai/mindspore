/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_UTILS_H_

#include <string>
#include <vector>
#include "kernel/kernel.h"
#include "./internal_kernel.h"

namespace mindspore {
namespace kernel {
class InternalKernelUtils {
 public:
  static internal::TensorFormat ToInternalFormat(Format format);
  static internal::TensorDType ToInternalDType(TypeId type);
  static int ToInternalOpId(std::string);
  static void ToInternalTensor(internal::Tensor *internal_tensor, const KernelTensor *kernel_tensor);

  static internal::DeviceRawBuf ToDeviceRawBuf(const KernelTensor *kernel_tensor);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_UTILS_H_
