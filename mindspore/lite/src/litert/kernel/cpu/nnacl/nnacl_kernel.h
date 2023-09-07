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
#include "src/executor/kernel_exec.h"
#include "src/litert/lite_kernel.h"

namespace mindspore::nnacl {
class NNACLKernel : public kernel::LiteKernel {
 public:
  explicit NNACLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                       const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    exec_env_ = const_cast<ExecEnv *>(ctx->GetExecEnv());
  }
  ~NNACLKernel() override;
  int Prepare() override;
  int ReSize() override;
  int Run() override;

  /* Execute after NNACLKernel creation
   * Create KernelBase */
  int InitKernel(const TypeId &data_type, const lite::InnerContext *ctx);
  const KernelBase *Kernel() const { return kernel_; }

 protected:
  void UpdateTensorC();
  int OptimizeDataCopy();

 private:
  int NNACLCheckArgs();

 protected:
  ExecEnv *exec_env_ = nullptr;
  KernelBase *kernel_ = nullptr;
  TensorC **in_ = nullptr;
  TensorC **out_ = nullptr;
  size_t in_size_ = 0;
  size_t out_size_ = 0;
};
}  // namespace mindspore::nnacl

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_NNACL_KERNEL_H_
