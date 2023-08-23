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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_DEFAULT_KERNEL_MOD_KERNEL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_DEFAULT_KERNEL_MOD_KERNEL_H_

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include "src/extendrt/kernel/base_kernel.h"
#include "kernel/kernel.h"
#include "ops/base_operator.h"

namespace mindspore::kernel {
class KernelModKernel : public BaseKernel {
 public:
  KernelModKernel(std::shared_ptr<mindspore::kernel::KernelMod> kernel_mod, BaseOperatorPtr base_operator,
                  CNodePtr cnode, const std::vector<InferTensor *> &in_tensors,
                  const std::vector<InferTensor *> &out_tensors, const InferContext *ctx)
      : BaseKernel({base_operator, cnode}, in_tensors, out_tensors, ctx),
        kernel_mod_(std::move(kernel_mod)),
        base_operator_(std::move(base_operator)),
        cnode_(std::move(cnode)) {}
  ~KernelModKernel() override = default;

  int Prepare() override;
  int InferShape() override;
  int ReSize() override;
  int Run() override;

 private:
  KernelModPtr kernel_mod_;
  BaseOperatorPtr base_operator_;
  CNodePtr cnode_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_DEFAULT_KERNEL_MOD_KERNEL_H_
