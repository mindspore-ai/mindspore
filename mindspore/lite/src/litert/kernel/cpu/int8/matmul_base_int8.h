/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_MATMUL_BASE_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_MATMUL_BASE_INT8_H_

#include <vector>
#include "include/errorcode.h"
#include "src/litert/lite_kernel.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/common_func.h"
#include "nnacl/int8/quantize.h"
#include "nnacl/int8/common_func_int8.h"
#include "nnacl/int8/matmul_int8.h"

namespace mindspore::kernel {
class MatmulBaseInt8CPUKernel : public LiteKernel {
  typedef void (*PackFunc)(const int8_t *src, int8_t *dst, int row, int col);

 public:
  MatmulBaseInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                          const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<MatMulParameter *>(op_parameter_);
    param_->matmul_type_ = MatmulType::kNotImplemented;
  }
  ~MatmulBaseInt8CPUKernel() override;
  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int MatmulReSize();

 public:
  int RunImpl(int task_id);
#if defined(ENABLE_ARM64) && !defined(SUPPORT_NNIE) && (!defined(MACHINE_LINUX_ARM64))
  int RunArm64Sdot();
  int Arm64SdotImpl(int task_id);
  int Arm64SdotPre(int task_id);
#endif

 protected:
  void InitParameter();

 private:
  void ResizeParameter();
  int InitBias();

 private:
  int InitTmpBuffer();
  void FreeTmpBuffer();
  int TransferB();

 private:
  int MallocQuantParam();
  int InitQuantParam();
  void FreeQuantParam();

 protected:
  MatMulParameter *param_ = nullptr;
  MatmulQuantParameter *quant_param_ = nullptr;
  int thread_count_ = 1;
  int thread_stride_ = 0;
  int8_t *pack_a_ptr_ = nullptr;
  int8_t *pack_b_ptr_ = nullptr;
  int *input_sums_ = nullptr;
  int *weight_bias_sums_ = nullptr;
  int *bias_ptr_ = nullptr;
  bool filter_per_channel_ = true;
  int8_t *batch_input_ptr_ = nullptr;
  int8_t *batch_weight_ptr_ = nullptr;
  int8_t *batch_b_ptr_ = nullptr;
  int8_t *batch_c_ptr_ = nullptr;
  int8_t *save_b_const_ = nullptr;
  int *batch_sums_ = nullptr;
  int row_tile_ = C4NUM;
  int col_tile_ = C4NUM;
  int deep_tile_ = C16NUM;
  int channel_num_ = 0;
  bool support_sdot_ = false;
  PackFunc a_pack_func_{nullptr};
  PackFunc b_pack_func_{nullptr};
  std::vector<int> a_offset_;
  std::vector<int> b_offset_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_MATMUL_BASE_INT8_H_
