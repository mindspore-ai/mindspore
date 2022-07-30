/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_CONCAT_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_CONCAT_FP16_H_

#include <arm_neon.h>
#include <vector>
#include "src/litert/kernel/cpu/base/concat_base.h"

using mindspore::lite::InnerContext;
namespace mindspore::kernel {
class ConcatFp16CPUKernel : public ConcatBaseCPUKernel {
 public:
  ConcatFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : ConcatBaseCPUKernel(parameter, inputs, outputs, ctx) {
    data_size_ = sizeof(float16_t);
  }
  ~ConcatFp16CPUKernel() = default;
  int Run() override;

 private:
  int EnsureFp16InputsAndOutput();
  std::vector<void *> tmp_buffers_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_CONCAT_FP16_H_
