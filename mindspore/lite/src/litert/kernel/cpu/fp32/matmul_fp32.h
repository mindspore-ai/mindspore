/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_MATMUL_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_MATMUL_FP32_H_

#include <vector>
#include "nnacl/matmul_parameter.h"
#include "src/litert/kernel/cpu/fp32/matmul_fp32_base.h"

namespace mindspore::kernel {
MatmulFp32BaseCPUKernel *CreateMatmulFp32CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                                   const std::vector<lite::Tensor *> &outputs,
                                                   const lite::InnerContext *ctx);
class MatmulCPUKernel : public LiteKernel {
 public:
  explicit MatmulCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                           const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    matmul_base_ = CreateMatmulFp32CPUKernel(parameter, inputs, outputs, ctx);
  }
  ~MatmulCPUKernel() {
    if (matmul_base_ != nullptr) {
      op_parameter_ = nullptr;  // op_parameter will be freed in LiteKernel
      delete matmul_base_;
      matmul_base_ = nullptr;
    }
  }

  void set_in_tensors(const std::vector<lite::Tensor *> &in_tensors) override {
    this->in_tensors_ = in_tensors;
    if (matmul_base_ != nullptr) {
      matmul_base_->set_in_tensors(in_tensors);
    }
  }

  void set_in_tensor(lite::Tensor *in_tensor, size_t index) override {
    MS_ASSERT(index < in_tensors_.size());
    this->in_tensors_[index] = in_tensor;
    if (matmul_base_ != nullptr) {
      matmul_base_->set_in_tensor(in_tensor, index);
    }
  }

  void set_out_tensors(const std::vector<lite::Tensor *> &out_tensors) override {
    this->out_tensors_ = out_tensors;
    if (matmul_base_ != nullptr) {
      matmul_base_->set_out_tensors(out_tensors);
    }
  }

  void set_out_tensor(lite::Tensor *out_tensor, size_t index) override {
    MS_ASSERT(index < out_tensors_.size());
    this->out_tensors_[index] = out_tensor;
    if (matmul_base_ != nullptr) {
      matmul_base_->set_out_tensor(out_tensor, index);
    }
  }

  // Train API
  int Train() override {
    (void)LiteKernel::Train();
    return matmul_base_->Train();
  }
  void SetTrainable(bool trainable) override {
    LiteKernel::SetTrainable(trainable);
    return matmul_base_->SetTrainable(trainable);
  }
  size_t workspace_size() override {
    (void)LiteKernel::workspace_size();
    return matmul_base_->workspace_size();
  }

  int Prepare() override;
  int ReSize() override;
  int Run() override;

  int PreparePackedWeight(const lite::Tensor *tensor) override;
  MatmulFp32BaseCPUKernel *GetMatmulBase() const { return matmul_base_; }

 private:
  MatmulFp32BaseCPUKernel *matmul_base_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_MATMUL_FP32_H_
