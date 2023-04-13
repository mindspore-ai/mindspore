/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP32_CUSTOM_GRU_FP32_H_
#define MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP32_CUSTOM_GRU_FP32_H_
#ifdef ENABLE_ARM64
#include <vector>
#include "src/litert/lite_kernel.h"

namespace mindspore::kernel {
class CustomGruCPUKernel : public LiteKernel {
 public:
  CustomGruCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                     const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {}
  ~CustomGruCPUKernel() override;
  int Prepare() override;
  int ReSize() override;
  int Run() override;

 private:
  int InitParamter();

 protected:
  void MallocRunBuffer(size_t data_type_size);
  virtual int InitWeightAndBias();
  int row_tile_{C12NUM};
  int col_tile_{C8NUM};
  void *weight_in_{nullptr};
  void *weight_hidden_{nullptr};
  void *bias_in_{nullptr};
  void *bias_hidden_{nullptr};
  void *init_h_{nullptr};
  void *run_buffer_{nullptr};
};
}  // namespace mindspore::kernel
#endif
#endif  // MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_FP32_CUSTOM_GRU_FP32_H_
