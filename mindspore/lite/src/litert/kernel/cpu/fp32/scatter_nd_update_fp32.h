/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP32_SCATTER_ND_UPDATE_FP32_H_
#define MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP32_SCATTER_ND_UPDATE_FP32_H_

#include <vector>
#include "src/litert/kernel/cpu/base/scatter_nd_binary.h"

namespace mindspore::kernel {

class ScatterNdUpdateCPUKernel : public ScatterNDBinaryCPUKernel {
 public:
  explicit ScatterNdUpdateCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                    const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : ScatterNDBinaryCPUKernel(parameter, inputs, outputs, ctx) {}
  ~ScatterNdUpdateCPUKernel() override = default;

  int Run() override;
  int ScatterNdUpdate(int task_id);
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP32_SCATTER_ND_UPDATE_FP32_H_
