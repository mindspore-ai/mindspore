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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_SLICE_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_SLICE_INT8_H_

#include <vector>
#include "src/runtime/kernel/arm/base/slice_base.h"
#include "mindspore/lite/nnacl/int8/quantize.h"

namespace mindspore::kernel {
class SliceInt8CPUKernel : public SliceCPUKernel {
 public:
  SliceInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                     const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : SliceCPUKernel(parameter, inputs, outputs, ctx) {}
  ~SliceInt8CPUKernel() = default;

  int Init() override;
  int Run() override;
  int DoSlice(int task_id);
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_SLICE_INT8_H_
