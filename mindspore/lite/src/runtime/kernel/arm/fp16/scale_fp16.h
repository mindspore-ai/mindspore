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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_SCALE_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_SCALE_FP16_H_

#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/fp32/scale_fp32.h"
#include "nnacl/scale.h"

namespace mindspore::kernel {

class ScaleFp16CPUKernel : public ScaleCPUKernel {
 public:
  ScaleFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                     const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : ScaleCPUKernel(parameter, inputs, outputs, ctx) {}
  ~ScaleFp16CPUKernel() = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int InitScaleOffset() override;
  int Scale(int task_id);

 private:
  int MallocAssignTmpBuffer();
  void FreeTmpBuffer();

 private:
  bool malloc_scale_ = false;
  bool malloc_offset_ = false;

  float16_t *input_ = nullptr;
  float16_t *scale_ = nullptr;
  float16_t *offset_ = nullptr;
  float16_t *output_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_SCALE_FP16_H_
