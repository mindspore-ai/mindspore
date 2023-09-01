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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_BISHENG_BISHENG_KERNEL_REGISTRY_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_BISHENG_BISHENG_KERNEL_REGISTRY_H_

#include <string>
#include <vector>
#include <memory>
#include "extendrt/delegate/type.h"
#include "extendrt/delegate/ascend_native/ascend_native_registration_factory.h"
#include "extendrt/delegate/ascend_native/ascend_native_base_kernel.h"

namespace mindspore::kernel {
template <class T>
AscendNativeBaseKernel *GetAscendNativeKernelOp(const std::vector<InferTensor *> &inputs,
                                                const std::vector<InferTensor *> &outputs, InferPrimitive prim,
                                                const InferContext *ctx, const void *stream, std::string name) {
  auto *op = new (std::nothrow) T(inputs, outputs, prim, ctx, stream, name);
  if (op == nullptr) {
    MS_LOG(WARNING) << "Ascend op is nullptr.";
    return nullptr;
  }
  return op;
}
typedef AscendNativeBaseKernel *(*AscendNativeKernelOp)(const std::vector<InferTensor *> &inputs,
                                                        const std::vector<InferTensor *> &outputs, InferPrimitive prim,
                                                        const InferContext *ctx, const void *stream, std::string name);

#define REGISTER_ASCEND_NATIVE_CREATOR(KEY, ASCEND_NATIVE_KERNEL_OP) \
  REGISTER_CLASS_CREATOR(std::string, KEY, AscendNativeKernelOp, GetAscendNativeKernelOp<ASCEND_NATIVE_KERNEL_OP>);

using AscendNativeRegistrationFactory = AutoRegistrationFactory<std::string, AscendNativeKernelOp>;
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_BISHENG_BISHENG_KERNEL_REGISTRY_H_
