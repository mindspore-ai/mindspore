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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_CARRY_DATA_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_CARRY_DATA_H_

#include <vector>
#include "src/lite_kernel.h"
#include "src/tensor.h"
#include "src/tensorlist.h"

namespace mindspore::kernel {
class CarryDataKernel : public LiteKernel {
 public:
  CarryDataKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                  const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {}
  ~CarryDataKernel() override = default;

 protected:
  int MoveData(std::vector<lite::Tensor *>::iterator dst_begin, std::vector<lite::Tensor *>::iterator dst_end,
               std::vector<lite::Tensor *>::iterator src_begin, std::vector<lite::Tensor *>::iterator src_limit);
  static int MoveTensorData(lite::Tensor *dst_tensor, lite::Tensor *src_tensor);
  static int MoveTensorListData(lite::TensorList *dst_tensor, lite::TensorList *src_tensor);
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_CARRY_DATA_H_
