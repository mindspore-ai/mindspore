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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_KERNEL_SELECTOR_H
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_KERNEL_SELECTOR_H

#include <vector>
#include <memory>
#include <string>
#include "src/infer/primitive_type.h"
#include "kernel/common_utils.h"
#include "src/infer/graph_compiler.h"
#include "src/extendrt/kernel/kernel_lib.h"
#include "src/infer/kernel.h"
#include "src/extendrt/graph_compiler/compile_option.h"

namespace mindspore::kernel {
class KernelSelector {
 public:
  explicit KernelSelector(const std::shared_ptr<lite::CompileOption> &compile_option)
      : compile_option_(compile_option) {}
  virtual ~KernelSelector() = default;
  virtual InferKernel *CreateKernel(const KernelSpec &spec, const std::vector<InferTensor *> &inputs,
                                    const std::vector<InferTensor *> &outputs, const InferContext *ctx) = 0;

 protected:
  // `format = DEFAULT_FORMAT` means not care about Format while select kernel.
  std::vector<const KernelLib *> Candidates(const PrimitiveType &op_type, const KernelAttr &require,
                                            const std::string &backend, Format format = DEFAULT_FORMAT);

 protected:
  const std::shared_ptr<lite::CompileOption> compile_option_{nullptr};
};

std::shared_ptr<KernelSelector> CreateKernelSelector(const std::shared_ptr<lite::CompileOption> &compile_option);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_KERNEL_SELECTOR_H
