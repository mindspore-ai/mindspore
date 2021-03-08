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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_CONSTANT_OF_SHAPE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_CONSTANT_OF_SHAPE_H_

#include <vector>
#include "include/errorcode.h"
#include "src/lite_kernel.h"
#include "include/context.h"
#include "nnacl/constant_of_shape_parameter.h"
#include "nnacl/fp32/constant_of_shape_fp32.h"
#include "nnacl/fp16/constant_of_shape_fp16.h"

namespace mindspore::kernel {
class ConstantOfShapeCPUKernel : public LiteKernel {
 public:
  ConstantOfShapeCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                           const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<ConstantOfShapeParameter *>(parameter);
  }
  ~ConstantOfShapeCPUKernel() override = default;

  int Init() override { return lite::RET_OK; }
  int ReSize() override { return lite::RET_OK; }
  int Run() override;
  int DoExecute(int task_id);

 private:
  ConstantOfShapeParameter *param_ = nullptr;
  void *output_ptr_ = nullptr;
  int thread_stride_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_CONSTANT_OF_SHAPE_H_
