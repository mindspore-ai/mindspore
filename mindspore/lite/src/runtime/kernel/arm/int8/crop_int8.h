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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_CROP_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_CROP_INT8_H_

#include <vector>
#include "src/lite_kernel.h"
#include "include/context.h"
#include "src/runtime/kernel/arm/base/crop_base.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::Context;

namespace mindspore::kernel {
class CropInt8CPUKernel : public CropBaseCPUKernel {
 public:
  CropInt8CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                    const std::vector<lite::tensor::Tensor *> &outputs, const Context *ctx,
                    const mindspore::lite::PrimitiveC *primitive)
      : CropBaseCPUKernel(parameter, inputs, outputs, ctx, primitive) {
    crop_para_ = reinterpret_cast<CropParameter *>(op_parameter_);
    crop_para_->thread_count_ = op_parameter_->thread_num_;
  }
  ~CropInt8CPUKernel() = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoExecute(int task_id);

 private:
  CropParameter *crop_para_;
};

int CropInt8Run(int task_id, LiteParallelGroupEnv *penv, void *cdata);
void PadOffset(int input_dim, CropParameter *crop_para);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_CROP_INT8_H_
