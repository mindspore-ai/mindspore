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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_NNACL_KERNEL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_NNACL_KERNEL_H_

#include <vector>
#include "nnacl/kernel.h"
#include "src/litert/kernel_exec.h"
#include "src/litert/lite_kernel.h"

namespace mindspore::nnacl {
class NnaclKernel : public kernel::LiteKernel {
 public:
  explicit NnaclKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                       const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {}
  ~NnaclKernel() override;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int InferShape();
  virtual int InitKernel(const kernel::KernelKey &key, const lite::InnerContext *ctx);

 protected:
  void UpdateTensorC();
  void UpdateTensorData();

 protected:
  KernelBase *kernel_ = nullptr;
  TensorC *in_ = nullptr;
  TensorC *out_ = nullptr;
  size_t in_size_ = 0;
  size_t out_size_ = 0;
};
}  // namespace mindspore::nnacl

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_NNACL_KERNEL_H_
