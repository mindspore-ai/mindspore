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
#include <string>
#include "src/extendrt/kernel/kernel_lib.h"
#include "src/extendrt/kernel/kernel_spec_infos.h"

namespace mindspore::kernel {
class DefaultKernelLib : public KernelLib {
 public:
  DefaultKernelLib() : KernelLib(kDefaultKernelLibName, kBackendCPU) {}

  bool Support(const PrimitiveType &op_type, const KernelAttr &attr, const std::string &backend,
               const Format &format = DEFAULT_FORMAT) const override;
  BaseKernel *CreateKernel(const KernelSpec &spec, const std::vector<InferTensor *> &inputs,
                           const std::vector<InferTensor *> &outputs, const InferContext *ctx) const override;
  InferKernel *CreateKernelExec(const KernelSpec &spec, const std::vector<InferTensor *> &inputs,
                                const std::vector<InferTensor *> &outputs, const InferContext *ctx) const override;

 private:
  static std::shared_ptr<mindspore::kernel::KernelMod> CreateKernelMod(const PrimitiveType &op_type,
                                                                       const KernelAttr &attr, const Format &format,
                                                                       const std::string &backend);
};
}  // namespace mindspore::kernel
#endif
