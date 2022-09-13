/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_TRANSPOSE_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_TRANSPOSE_BASE_H_

#include <vector>
#include "nnacl/transpose.h"
#include "src/litert/lite_kernel.h"

namespace mindspore::kernel {
class TransposeBaseCPUKernel : public LiteKernel {
 public:
  explicit TransposeBaseCPUKernel(OpParameter *param, const std::vector<lite::Tensor *> &inputs,
                                  const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(param, inputs, outputs, ctx) {
    param_ = reinterpret_cast<TransposeParameter *>(param);
  }
  ~TransposeBaseCPUKernel() override = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  virtual int DoTransposeMultiThread(int task_id) = 0;

 protected:
  // only true when perm is [1, 0] or [0, 2, 1]
  bool opt_run_{true};

  // we suppose that the transpose is invalid when out-data is equal to in-data, and others need to do explicit
  // data-transpose.
  bool is_valid_{true};

  void *in_data_ = nullptr;

  void *out_data_ = nullptr;

  // only valid when opt_run_ is true
  int opt_param_[3] = {0};

  int out_shape_[MAX_TRANSPOSE_DIM_SIZE] = {0};

  TransposeParameter *param_{nullptr};

 private:
  virtual int DoTransposeSingleThread() = 0;
  int CopyInputToOutput();
  int ResetStatus();
  // to simplify transpose, we consider two steps. Firstly, delete the dimension where the value is 1. Secondly, fuse
  // continuous dimensions. The perm will be updated along with the data-shape. Example, data-shape is [2, 1, 3, 4, 5]
  // and perm is [0, 4, 2, 3, 1]. After first step, data-shape is [2, 3, 4, 5] and perm is [0, 3, 1, 2]. After second
  // step, data-shape is [2, 12, 5] and perm is [0, 2, 1]. We confirm that these transformations are equivalent.
  int OptimizeShape();

  // if perm is [0, 1] or [0, 2, 1], we will do transpose-opt.
  void SetTransposeOptInfo();

  int ComputeOfflineInfo();

  // optimized shape
  std::vector<int> in_shape_;

  // optimized perm
  std::vector<int> perm_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_TRANSPOSE_BASE_H_
