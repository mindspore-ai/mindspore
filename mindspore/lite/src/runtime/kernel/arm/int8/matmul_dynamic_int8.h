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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_MATMUL_DYNAMIC_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_MATMUL_DYNAMIC_INT8_H_

#include <vector>
#include "include/errorcode.h"
#include "include/context.h"
#include "src/inner_kernel.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/common_func.h"
#include "nnacl/int8/quantize.h"
#include "nnacl/int8/common_func_int8.h"
#include "nnacl/int8/matmul_int8.h"

namespace mindspore::kernel {
class MatmulDynamicInt8CPUKernel : public InnerKernel {
  typedef void (*PackFunc)(const int8_t *src, int8_t *dst, int row, int col);

 public:
  MatmulDynamicInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                             const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<MatMulParameter *>(op_parameter_);
  }
  ~MatmulDynamicInt8CPUKernel() override;
  int Prepare() override;
  int ReSize() override;
  int Run() override;

 public:
  int RunImpl(int task_id);
#if defined(ENABLE_ARM64) && !defined(SUPPORT_NNIE) && (!defined(MACHINE_LINUX_ARM64))
  int RunArm64Sdot();
  int Arm64SdotImpl(int task_id);
  int Arm64SdotPre(int task_id);
#endif

 private:
  void InitParameter();
  void ResizeParameter();
  int CopyBias();
  int InitTmpBuffer();
  void FreeTmpBuffer();
  int TransferA();
  int TransferB();

  int MallocQuantParam();
  int InitInputQuantParam();
  int InitFilterQuantParam();
  void FreeQuantParam();

 private:
  MatMulParameter *param_ = nullptr;
  MatmulDynamicQuantParameter *quant_param_ = nullptr;
  int thread_count_ = 1;
  int thread_stride_ = 0;
  int8_t *pack_a_ptr_ = nullptr;
  int8_t *pack_b_ptr_ = nullptr;
  float *fp32_bias_ptr_ = nullptr;
  bool filter_per_channel_ = true;
  int8_t *batch_input_ptr_ = nullptr;
  int8_t *batch_weight_ptr_ = nullptr;
  int8_t *batch_b_ptr_ = nullptr;
  float *batch_c_ptr_ = nullptr;
  int row_tile_ = C4NUM;
  int col_tile_ = C4NUM;
  int deep_tile_ = C16NUM;
  int channel_num_ = 0;
  bool support_sdot_ = false;
  PackFunc a_pack_func_{nullptr};
  PackFunc b_pack_func_{nullptr};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_MATMUL_DYNAMIC_INT8_H_
