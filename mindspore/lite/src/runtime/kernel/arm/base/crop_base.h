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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_CROP_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_CROP_BASE_H_

#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/crop_parameter.h"

namespace mindspore::kernel {
class CropBaseCPUKernel : public LiteKernel {
 public:
  CropBaseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                    const std::vector<lite::Tensor *> &outputs, const mindspore::lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    crop_para_ = reinterpret_cast<CropParameter *>(op_parameter_);
    crop_para_->thread_count_ = op_parameter_->thread_num_;
  }
  ~CropBaseCPUKernel() = default;

  int Init() override;
  int ReSize() override;
  int Run() override { return 0; }

 protected:
  std::vector<int> input_shape_;
  std::vector<int> output_shape_;
  CropParameter *crop_para_;
  void PadOffset(int input_dim, CropParameter *crop_para);
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_CROP_BASE_H_
