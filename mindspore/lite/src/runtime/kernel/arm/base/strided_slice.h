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

#ifndef MINDSPORE_LITE_SRC_BACKEND_ARM_BASE_STRIDED_SLICE_H_
#define MINDSPORE_LITE_SRC_BACKEND_ARM_BASE_STRIDED_SLICE_H_

#include <vector>
#include "ir/anf.h"
#include "src/lite_kernel.h"

namespace mindspore::kernel {
class StridedSliceCPUKernel : public LiteKernel {
 public:
  StridedSliceCPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                        const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                        const mindspore::lite::PrimitiveC *primitive)
      : LiteKernel(parameter, inputs, outputs, ctx, primitive) {}
  ~StridedSliceCPUKernel() override = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_BACKEND_ARM_BASE_STRIDED_SLICE_H_
