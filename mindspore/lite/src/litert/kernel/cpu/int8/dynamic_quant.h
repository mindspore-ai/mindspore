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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_DYNAMIC_QUANT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_DYNAMIC_QUANT_H_

#include <vector>
#include <cfloat>
#include "src/litert/lite_kernel.h"

namespace mindspore::kernel {
class DynamicQuantCPUKernel : public LiteKernel {
 public:
  DynamicQuantCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                        const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx), thread_num_(ctx->thread_num_) {}
  ~DynamicQuantCPUKernel() override = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;

  int QuantData(int task_id);
  int CalculateMinMax(int task_id);

 private:
  void ReduceMinMaxFp32();
  void CalculateScaleZp();

 private:
  int thread_num_;
  int thread_n_num_{0};
  int thread_n_stride_{0};
  int num_unit_{0};
  int8_t *int8_ptr_ = nullptr;
  float *float32_ptr_ = nullptr;

  float real_min_array_[8] = {0};
  float real_max_array_[8] = {0};
  float real_min_ = FLT_MAX;
  float real_max_ = FLT_MIN;
  int32_t src_dtype_{0};
  int32_t dst_dtype_{0};
  bool symmetric_ = false;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_DYNAMIC_QUANT_H_
