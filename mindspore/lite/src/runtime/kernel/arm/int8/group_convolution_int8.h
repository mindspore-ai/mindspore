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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_GROUP_CONVOLUTION_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_GROUP_CONVOLUTION_INT8_H_

#include <utility>
#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/op_base.h"
#include "src/runtime/kernel/arm/fp32/group_convolution_fp32.h"

namespace mindspore::kernel {
class GroupConvolutionInt8CPUKernel : public GroupConvolutionCPUKernel {
 public:
  GroupConvolutionInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                                std::vector<kernel::LiteKernel *> group_convs, const int group_num)
      : GroupConvolutionCPUKernel(parameter, inputs, outputs, ctx, std::move(group_convs), group_num) {
  }  // opParameter(in channel, out channel) in this kernel has been split to groups, if
  // you want to get real params, multiply in channel / out channel with group num

  int Run() override;
  void SeparateInput(int group_id) override;
  void PostConcat(int group_id) override;

 private:
  int8_t *ori_in_data_ = nullptr;   // do not free
  int8_t *ori_out_data_ = nullptr;  // do not free
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_GROUP_CONVOLUTION_INT8_H_
