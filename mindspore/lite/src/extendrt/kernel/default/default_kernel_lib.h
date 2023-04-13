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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_DEFAULT_DEFAULT_LIB_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_DEFAULT_DEFAULT_LIB_H_

#include <vector>
#include <memory>
#include "src/extendrt/kernel/kernel_lib.h"

using mindspore::infer::abstract::Tensor;

namespace mindspore::kernel {
constexpr char kDefaultKernelLibName[] = "Default";
class DefaultKernelLib : public KernelLib {
 public:
  DefaultKernelLib() : KernelLib(kDefaultKernelLibName, "CPU") {}

  bool Support(const PrimitiveType &op_type, const KernelAttr &attr, const Format &format) override;
  LiteKernel *CreateKernel(const KernelSpec &spec, const std::vector<InferTensor *> &inputs,
                           const std::vector<InferTensor *> &outputs, const InferContext *ctx) override;

 private:
  static std::shared_ptr<mindspore::kernel::KernelMod> CreateKernelMod(const PrimitiveType &op_type,
                                                                       const KernelAttr &attr, const Format &format);
};
}  // namespace mindspore::kernel
#endif
