/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_BASE_SCATTER_ND_BINARY_H_
#define MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_BASE_SCATTER_ND_BINARY_H_

#include <vector>
#include "src/litert/lite_kernel.h"
#include "nnacl/base/scatter_nd_binary.h"

namespace mindspore::kernel {
constexpr int kScatterUpdateInputIndex = 0;
constexpr int kScatterIndicesIndex = 1;
constexpr int kScatterUpdateIndex = 2;

class ScatterNDBinaryCPUKernel : public LiteKernel {
 public:
  explicit ScatterNDBinaryCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                    const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<ScatterNDParameter *>(parameter);
  }
  ~ScatterNDBinaryCPUKernel() override = default;

  int Prepare() override;
  int ReSize() override;

 protected:
  ScatterNDParameter *param_ = nullptr;
  std::vector<int> output_unit_offsets_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_BASE_SCATTER_ND_BINARY_H_
