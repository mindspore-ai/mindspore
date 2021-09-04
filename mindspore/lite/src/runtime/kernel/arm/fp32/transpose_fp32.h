/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_KERNEL_CPU_ARM_FP32_TRANSPOSE_H_
#define MINDSPORE_CCSRC_KERNEL_CPU_ARM_FP32_TRANSPOSE_H_

#include <vector>
#include "include/errorcode.h"
#include "nnacl/fp32/transpose_fp32.h"
#include "nnacl/transpose.h"
#include "src/inner_kernel.h"
#include "src/kernel_registry.h"

namespace mindspore::kernel {

typedef void (*TransposeFunc)(const void *src, void *dst, int batch, int plane, int channel, int thread_num,
                              int task_id);

class TransposeCPUKernel : public InnerKernel {
 public:
  explicit TransposeCPUKernel(OpParameter *param, const std::vector<lite::Tensor *> &inputs,
                              const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(param, inputs, outputs, ctx) {
    param_ = reinterpret_cast<TransposeParameter *>(param);
  }
  ~TransposeCPUKernel() override;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id);

 protected:
  virtual void GetNchwToNhwcFunc(TypeId dtype);
  virtual void GetNhwcToNchwFunc(TypeId dtype);
  virtual int TransposeDim2to6();
  virtual int TransposeDimGreaterThan6(int task_id);

  int GetNHNCTransposeFunc(const lite::Tensor *in_tensor, const lite::Tensor *out_tensor);
  void *in_data_ = nullptr;
  void *out_data_ = nullptr;
  int *out_shape_ = nullptr;
  TransposeParameter *param_ = nullptr;
  TransposeFunc NHNCTransposeFunc_ = nullptr;
  int nhnc_param_[3] = {0};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_CCSRC_KERNEL_CPU_ARM_FP32_TRANSPOSE_H_
