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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GROUP_CONVOLUTION_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GROUP_CONVOLUTION_H_

#include <utility>
#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/op_base.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"
#include "nnacl/fp32/conv.h"

namespace mindspore::kernel {
class GroupConvolutionCPUKernel : public ConvolutionBaseCPUKernel {
 public:
  GroupConvolutionCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                            const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                            const mindspore::lite::PrimitiveC *primitive, std::vector<kernel::LiteKernel *> group_convs,
                            const int group_num)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx, primitive),
        group_convs_(std::move(group_convs)),
        group_num_(group_num) {}  // opParameter(in channel, out channel) in this kernel has been split to groups, if
                                  // you want to get real params, multiply in channel / out channel with group num
  ~GroupConvolutionCPUKernel() override {
    for (auto sub_conv : group_convs_) {
      // free sub conv input tensors / output tensors manually
      auto sub_in_tensors = sub_conv->in_tensors();
      auto sub_in_tensor_num = sub_in_tensors.size();
      for (size_t i = 0; i < sub_in_tensor_num; ++i) {
        delete sub_in_tensors[i];
      }
      auto sub_out_tensors = sub_conv->out_tensors();
      auto sub_out_tensor_num = sub_out_tensors.size();
      for (size_t i = 0; i < sub_out_tensor_num; ++i) {
        delete sub_out_tensors[i];
      }
      delete sub_conv;
    }
  };

  int Init() override;
  int ReSize() override;
  int Run() override;
  int PreProcess() override;
  void SeparateInput(int group_id);
  void PostConcat(int group_id);

 private:
  std::vector<kernel::LiteKernel *> group_convs_;
  float *ori_in_data_ = nullptr;   // do not free
  float *ori_out_data_ = nullptr;  // do not free
  const int group_num_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GROUP_CONVOLUTION_H_
