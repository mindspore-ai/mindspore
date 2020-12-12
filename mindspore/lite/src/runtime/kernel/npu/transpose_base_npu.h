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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_TRANSPOSE_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_TRANSPOSE_BASE_H_

#include <vector>
#include "include/graph/op/all_ops.h"
#include "include/graph/compatible/all_ops.h"
#include "src/runtime/kernel/npu/npu_kernel.h"
#include "nnacl/op_base.h"

namespace mindspore::kernel {
class TransposeBaseNPUKernel : public NPUKernel {
 public:
  TransposeBaseNPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                         const mindspore::lite::PrimitiveC *primitive)
      : NPUKernel(parameter, inputs, outputs, ctx, primitive) {}
  ~TransposeBaseNPUKernel() override;

 protected:
  int SetPreTranspose(const ge::Operator *input);
  int SetPostTranspose(const ge::Operator *input);
  hiai::op::Permute *pre_trans_ = nullptr;
  hiai::op::Permute *post_trans_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_TRANSPOSE_BASE_H_
