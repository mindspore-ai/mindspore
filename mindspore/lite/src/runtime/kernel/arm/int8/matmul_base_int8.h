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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_MATMUL_BASE_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_MATMUL_BASE_INT8_H_

#include <vector>
#include "include/errorcode.h"
#include "include/context.h"
#include "src/lite_kernel.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/common_func.h"
#include "nnacl/int8/quantize.h"
#include "nnacl/int8/common_func_int8.h"
#include "nnacl/int8/matmul_int8.h"

namespace mindspore::kernel {
class MatmulBaseInt8CPUKernel : public LiteKernel {
 public:
  MatmulBaseInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                          const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<MatMulParameter *>(op_parameter_);
  }
  ~MatmulBaseInt8CPUKernel() override;
  int Init() override;
  int ReSize() override;
  int Run() override;

 public:
  int RunImpl(int task_id);

 protected:
  void InitParameter();

 private:
  void ResizeParameter();
  int InitBias();

 private:
  int InitTmpBuffer();
  void FreeTmpBuffer();
  void TransferA();
  void TransferB();

 private:
  int MallocQuantParam();
  void FreeQuantParam();
  void InitQuantParam();

 protected:
  MatMulParameter *param_ = nullptr;
  MatmulQuantParameter quant_;
  int thread_count_ = 1;
  int thread_stride_ = 0;
  int8_t *pack_a_ptr_ = nullptr;
  int8_t *pack_b_ptr_ = nullptr;
  int *input_sums_ = nullptr;
  int *weight_bias_sums_ = nullptr;
  int *bias_ptr_ = nullptr;
  bool filter_per_channel_ = true;
  int8_t *batch_b_ptr_ = nullptr;
  int8_t *batch_c_ptr_ = nullptr;
  int *batch_sums_ = nullptr;
  int row_tile_ = C4NUM;
  int col_tile_ = C4NUM;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_MATMUL_BASE_INT8_H_
