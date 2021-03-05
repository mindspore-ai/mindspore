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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_BATCH_TO_SPACE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_BATCH_TO_SPACE_H_

#include <vector>
#include "include/errorcode.h"
#include "nnacl/batch_to_space.h"
#include "nnacl/base/batch_to_space_base.h"
#include "src/lite_kernel.h"

namespace mindspore::kernel {
class BatchToSpaceCPUKernel : public LiteKernel {
 public:
  BatchToSpaceCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                        const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {}
  ~BatchToSpaceCPUKernel() = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int Processinput();

 private:
  int32_t block_shape_[BATCH_TO_SPACE_BLOCK_SHAPE_SIZE];
  int32_t crops_[COMM_SHAPE_SIZE];
  bool no_crop_ = false;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_BATCH_TO_SPACE_H_
