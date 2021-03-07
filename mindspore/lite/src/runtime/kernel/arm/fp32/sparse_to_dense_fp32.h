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
#include "mindspore/lite/nnacl/fp32/sparse_to_dense_fp32.h"
#include "src/runtime/kernel/arm/base/layout_transform.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class SparseToDenseCPUKernel : public LiteKernel {
 public:
  SparseToDenseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx), ctx_(ctx), thread_count_(ctx->thread_num_) {
    s2d_param = (reinterpret_cast<SparseToDenseParameter *>(op_parameter_));
    s2d_param->thread_num_ = thread_count_;
  }
  ~SparseToDenseCPUKernel() = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoExcute(int task_id);
  int GenerateIndices();
  int IndicesValidCheck();

 protected:
  const InnerContext *ctx_;
  int thread_count_;
  SparseToDenseParameter *s2d_param;

 private:
  int **sparse_indices_vect = nullptr;
  float *sparse_values = nullptr;
  float default_value;
  bool isScalar = false;
  int index_num;
  int index_dim;
  float *output_data = nullptr;
  int output_shape[4];
  int output_num;
  int64_t count_unit_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_SPARSETODENSE_H_
