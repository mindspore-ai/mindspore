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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_PAD_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_PAD_INT8_H_

#include <string>
#include <vector>
#include "include/errorcode.h"
#include "src/lite_kernel.h"
#include "src/runtime/runtime_api.h"
#include "nnacl/errorcode.h"
#include "nnacl/pad_parameter.h"
#include "nnacl/int8/pad_int8.h"
#include "nnacl/int8/quantize.h"

namespace mindspore::kernel {
class PadInt8CPUKernel : public LiteKernel {
 public:
  explicit PadInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                            const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    op_parameter_->thread_num_ = ctx->thread_num_;
    pad_param_ = reinterpret_cast<PadParameter *>(op_parameter_);
  }
  ~PadInt8CPUKernel() override { FreeQuantParam(); };

  int Init() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id);
  int RunMirrorPadImpl(int task_id);

 private:
  int SetQuantParam();
  int InitPadParam();
  void FreeQuantParam();

 private:
  int HandleMirrorPad();
  int CheckPaddings(const int *paddings, int length, const int *input_shape, int mode);
  int CopyPaddingFromInput();
  void CalculateStrides();
  int ExtendPaddings(int *paddings, int length, const int *ori_paddings, int ori_length);

  PadParameter *pad_param_ = nullptr;
  int8_t *in_data_ = nullptr;
  int8_t *out_data_ = nullptr;
  int in_dims_[COMM_SHAPE_SIZE] = {0};
  int out_dims_[COMM_SHAPE_SIZE] = {0};
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_PAD_INT8_H_
