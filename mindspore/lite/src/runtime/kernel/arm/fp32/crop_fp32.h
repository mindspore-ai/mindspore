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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CROP_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CROP_H_

#include <vector>
#include "include/errorcode.h"
#include "nnacl/fp32/crop_fp32.h"
#include "nnacl/crop_parameter.h"
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/base/layout_transform.h"
#include "src/runtime/kernel/arm/base/crop_base.h"

namespace mindspore::kernel {
class CropCPUKernel : public CropBaseCPUKernel {
 public:
  CropCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : CropBaseCPUKernel(parameter, inputs, outputs, ctx) {}
  ~CropCPUKernel() = default;
  int Init() override;
  int Run() override;
  int CropParallelRun(int thread_id);
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CROP_H_
