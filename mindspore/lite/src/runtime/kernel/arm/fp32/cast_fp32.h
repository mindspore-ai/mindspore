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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CAST_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CAST_H_

#include <vector>
#include "include/errorcode.h"
#include "src/inner_kernel.h"
#include "src/tensor.h"
#include "nnacl/op_base.h"
#include "nnacl/base/cast_base.h"

namespace mindspore::kernel {
class CastCPUKernel : public InnerKernel {
 public:
  CastCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {}

  ~CastCPUKernel() = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoCast(int thread_id);

 private:
  int CastToFp32(const lite::Tensor *input, lite::Tensor *output, int offset, int data_num);
  int CastToFp16(const lite::Tensor *input, lite::Tensor *output, int offset, int data_num);
  int CastToOthers(const lite::Tensor *input, lite::Tensor *output, int offset, int data_num);
  int stride_ = 0;
  int data_num_ = 0;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CAST_H_
