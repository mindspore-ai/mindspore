/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_SCATTER_ND_UPDATE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_SCATTER_ND_UPDATE_H_

#include <vector>
#include "src/inner_kernel.h"

namespace mindspore::kernel {

class ScatterNdUpdateCPUKernel : public InnerKernel {
 public:
  explicit ScatterNdUpdateCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                    const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {}
  ~ScatterNdUpdateCPUKernel() override = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int ScatterNdUpdate(int task_id);

 private:
  int thread_n_num_ = 1;
  int thread_n_stride_ = 1;
  int num_unit_ = 1;
  int unit_size_ = 1;
  float *output_ptr_ = nullptr;
  float *update_ptr_ = nullptr;
  std::vector<int> out_strides_;
  std::vector<int> output_unit_offsets_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_SCATTER_ND_UPDATE_H_
