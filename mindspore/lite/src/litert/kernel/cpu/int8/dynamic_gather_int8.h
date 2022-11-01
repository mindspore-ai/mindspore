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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_DYNAMIC_GATHER_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_DYNAMIC_GATHER_INT8_H_

#include <vector>
#include "nnacl/gather_parameter.h"
#include "nnacl/int8/quantize.h"
#include "src/litert/lite_kernel.h"

namespace mindspore::kernel {
class DynamicGatherInt8CPUKernel : public LiteKernel {
 public:
  DynamicGatherInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                             const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx), thread_count_(ctx->thread_num_) {}
  ~DynamicGatherInt8CPUKernel() override;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoGather(int task_id);

 private:
  int AssignIndicesData(bool isIndicesInt32, int indices_num, lite::Tensor *indices_tensor, int limit);

 private:
  int thread_count_ = 0;
  int inner_size_ = 0;
  int limit_ = 0;
  int outer_size_ = 0;
  int axis_ = 0;
  int indices_element_size_ = 0;
  int *indices_data_ = nullptr;
  bool enable_fp16_ = false;
  DynamicGatherQuantArg *quant_param_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_DYNAMIC_GATHER_INT8_H_
