/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_EXTENDRT_KERNEL_EXEC_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_EXTENDRT_KERNEL_EXEC_H_
#include <utility>
#include <memory>
#include <vector>
#include "src/executor/kernel_exec.h"
#include "src/extendrt/kernel/base_kernel.h"

namespace mindspore::kernel {
class ExtendRTKernelExec : public KernelExec {
 public:
  ExtendRTKernelExec() : KernelExec() {}

  explicit ExtendRTKernelExec(std::shared_ptr<MSKernel> kernel) : KernelExec(std::move(kernel)) {}

  ~ExtendRTKernelExec() override = default;

  bool IsBuiltin() override {
    if (desc_.provider != kBuiltin) {
      MS_LOG(EXCEPTION) << "Custom kernel not supported in ExtendRT now.";
    }
    return false;
  }

  OpParameter *op_parameter() const override {
    MS_ASSERT(kernel_ != nullptr);
    return std::static_pointer_cast<BaseKernel>(kernel_)->op_parameter();
  }

  PrimitiveType type() const override { return reinterpret_cast<BaseKernel *>(kernel_.get())->type(); }

  void set_in_tensors(const std::vector<lite::Tensor *> &in_tensors) override {
    std::static_pointer_cast<BaseKernel>(kernel_)->set_in_tensors(in_tensors);
  }

  void set_in_tensor(lite::Tensor *in_tensor, size_t index) override {
    std::static_pointer_cast<BaseKernel>(kernel_)->set_in_tensor(in_tensor, index);
  }

  void set_out_tensors(const std::vector<lite::Tensor *> &out_tensors) override {
    std::static_pointer_cast<BaseKernel>(kernel_)->set_out_tensors(out_tensors);
  }

  void set_out_tensor(lite::Tensor *out_tensor, size_t index) override {
    std::static_pointer_cast<BaseKernel>(kernel_)->set_out_tensor(out_tensor, index);
  }

  const std::vector<lite::Tensor *> &in_tensors() const override {
    return std::static_pointer_cast<BaseKernel>(kernel_)->in_tensors();
  }

  const std::vector<lite::Tensor *> &out_tensors() const override {
    return std::static_pointer_cast<BaseKernel>(kernel_)->out_tensors();
  }
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_EXTENDRT_KERNEL_EXEC_H_
