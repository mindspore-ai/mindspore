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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_SPARSE_FILL_EMPTY_ROWS_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_SPARSE_FILL_EMPTY_ROWS_FP32_H_

#include <vector>
#include "src/litert/lite_kernel.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class SparseFillEmptyRowsCPUKernel : public LiteKernel {
 public:
  SparseFillEmptyRowsCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                               const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    // sparse_fill_empty_rows_param_ = reinterpret_cast<SparseFillEmptyRowsParameter *>(op_parameter_);
  }
  ~SparseFillEmptyRowsCPUKernel() = default;

  int PreProcess() override;
  int Prepare() override;
  int ReSize() override { return RET_OK; }
  int Run() override;
  // virtual int DoExcute(int task_id);

 private:
  void UpdataTensorShape(lite::Tensor *tensor, std::vector<int> *new_shape);
  int RunInferOutputShape();
  template <typename T>
  int RunOutputData();

 protected:
  // SparseFillEmptyRowsParameter *sparse_fill_empty_rows_param_;
  std::vector<int32_t> scratch_;
  int32_t dense_rows_ = 0;
  int32_t N_ = 0;
  int32_t rank_ = 0;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_SPARSE_FILL_EMPTY_ROWS_FP32_H_
