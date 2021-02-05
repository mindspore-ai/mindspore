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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_TRANSPOSE_NPU_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_TRANSPOSE_NPU_H_
#include <vector>
#include "include/graph/op/all_ops.h"
#include "nnacl/transpose.h"
#include "src/runtime/kernel/npu/npu_kernel.h"
namespace mindspore::kernel {
class TransposeNPUKernel : public NPUKernel {
 public:
  TransposeNPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                     const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                     const mindspore::lite::PrimitiveC *primitive)
      : NPUKernel(parameter, inputs, outputs, ctx, primitive) {
    if (primitive->Type() == schema::PrimitiveType_Transpose) {
      auto transpose_parameter = reinterpret_cast<TransposeParameter *>(parameter);
      conjugate_ = transpose_parameter->conjugate_;
      for (int i = 0; i < transpose_parameter->num_axes_; i++) {
        perm_.push_back(transpose_parameter->perm_[i]);
      }
    } else if (primitive->Type() == schema::PrimitiveType_Nchw2Nhwc) {
      perm_ = {0, 2, 3, 1};
    } else if (primitive->Type() == schema::PrimitiveType_Nhwc2Nchw) {
      perm_ = {0, 3, 1, 2};
    }
  }
  ~TransposeNPUKernel() override;

  int IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                OpParameter *opParameter) override;
  int SetNPUInputs(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                   const std::vector<ge::Operator *> &npu_inputs) override;
  ge::Operator *GetNPUOp() override;

 private:
  hiai::op::Permute *op_ = nullptr;
  std::vector<int64_t> perm_;
  bool conjugate_ = false;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_TRANSPOSE_NPU_H_
