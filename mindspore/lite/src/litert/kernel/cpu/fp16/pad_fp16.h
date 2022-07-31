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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_PAD_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_PAD_FP16_H_

#include <vector>
#include "src/litert/kernel/cpu/fp32/pad_fp32.h"
#include "nnacl/fp16/pad_fp16.h"

namespace mindspore::kernel {
class PadFp16CPUKernel : public PadCPUKernel {
 public:
  PadFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                   const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : PadCPUKernel(parameter, inputs, outputs, ctx) {}

  ~PadFp16CPUKernel() {}

  int Run() override;
  int RunImpl(int task_id) const override;
  int RunMirrorPadImpl(int task_id) const override;

 private:
  void RunMirrorPadImplFast(const MirrorPadBlock &block, const float16_t *input_data, float16_t *output_data) const;

 private:
  float16_t *input_ = nullptr;
  float16_t *output_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_PAD_FP16_H_
