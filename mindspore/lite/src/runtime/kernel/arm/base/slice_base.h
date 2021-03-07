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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_SLICE_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_SLICE_BASE_H_

#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/slice_parameter.h"

namespace mindspore::kernel {
class SliceCPUKernel : public LiteKernel {
 public:
  SliceCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                 const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<SliceParameter *>(op_parameter_);
  }
  ~SliceCPUKernel() = default;

  int Init() override;
  int ReSize() override;
  int Run() override;

 public:
  int SliceParallelRun(int thread_id);

 protected:
  SliceParameter *param_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_SLICE_BASE_H_
