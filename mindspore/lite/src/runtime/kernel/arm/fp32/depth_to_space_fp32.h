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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_DEPTH_TO_SPACE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_DEPTH_TO_SPACE_H_

#include <vector>
#include "include/errorcode.h"
#include "nnacl/base/depth_to_space_base.h"
#include "src/runtime/kernel/arm/base/depth_to_space_base.h"

namespace mindspore::kernel {
class DepthToSpaceCPUKernel : public DepthToSpaceBaseCPUKernel {
 public:
  DepthToSpaceCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                        const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : DepthToSpaceBaseCPUKernel(parameter, inputs, outputs, ctx) {}
  ~DepthToSpaceCPUKernel() = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_DEPTH_TO_SPACE_H_
