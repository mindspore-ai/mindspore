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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_FULLCONNECTION_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_FULLCONNECTION_INT8_H_

#include <vector>
#include "src/lite_kernel.h"
#include "include/errorcode.h"
#include "mindspore/lite/nnacl/int8/quantize.h"
#include "nnacl/common_func.h"
#include "nnacl/int8/common_func_int8.h"
#include "nnacl/int8/matmul_int8.h"

namespace mindspore::kernel {
class FullconnectionInt8CPUKernel : public LiteKernel {
 public:
  FullconnectionInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                              const std::vector<lite::Tensor *> &outputs, const mindspore::lite::InnerContext *ctx,
                              const mindspore::lite::PrimitiveC *primitive)
      : LiteKernel(parameter, inputs, outputs, ctx, primitive) {
    fc_param_ = reinterpret_cast<MatMulParameter *>(op_parameter_);
  }
  ~FullconnectionInt8CPUKernel() override {
    FreeTmpBuffer();
    FreeQuantParam();
  }

  int Init() override;
  int ReSize() override;
  int Run() override;

 public:
  int RunImpl(int task_id);

 private:
  void InitParam();
  void FreeTmpBuffer();
  void FreeQuantParam();
  int MallocQuantParam();

 private:
  MatMulParameter *fc_param_ = nullptr;
  MatmulQuantParameter quant_;
  int thread_count_ = 1;
  int thread_stride_ = 0;
  int8_t *pack_a_ptr_ = nullptr;
  int8_t *pack_b_ptr_ = nullptr;
  int8_t *c_ptr_ = nullptr;
  int *input_sums_ = nullptr;
  int *weight_bias_sums_ = nullptr;
  int *bias_ptr_ = nullptr;
  bool filter_per_channel_ = true;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_FULLCONNECTION_INT8_H_
