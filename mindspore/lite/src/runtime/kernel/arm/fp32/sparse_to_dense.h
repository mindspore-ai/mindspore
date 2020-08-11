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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_SPARSETODENSE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_SPARSETODENSE_H_

#include <vector>
#include "src/lite_kernel.h"

#include "include/context.h"
#include "src/runtime/kernel/arm/nnacl/sparse_to_dense.h"
#include "src/runtime/kernel/arm/base/layout_transform.h"

using mindspore::lite::Context;

namespace mindspore::kernel {
class SparseToDenseCPUKernel : public LiteKernel {
 public:
  SparseToDenseCPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                         const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                         const lite::Primitive *primitive)
      : LiteKernel(parameter, inputs, outputs, ctx, primitive), ctx_(ctx), thread_count_(ctx->thread_num_) {
    s2d_param_ = (reinterpret_cast<SparseToDenseParameter *>(op_parameter_));
  }
  ~SparseToDenseCPUKernel() = default;

  int Init() override;
  int ReSize() override { return 0; }
  int Run() override;
  int DoExcute(int task_id);

 protected:
  int thread_count_;
  const Context *ctx_;
  SparseToDenseParameter *s2d_param_;

 private:
  int *input_data_;
  int *total_number_;
  int sp_num_;
  float *snum_;
  float *dnum_;
  float *output_data;
  int *output_shape_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_SPARSETODENSE_H_
