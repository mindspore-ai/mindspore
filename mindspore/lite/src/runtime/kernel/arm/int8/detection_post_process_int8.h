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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_DETECTION_POST_PROCESS_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_DETECTION_POST_PROCESS_INT8_H_

#include <vector>
#include "src/lite_kernel.h"
#include "include/context.h"
#include "src/runtime/kernel/arm/base/detection_post_process_base.h"
#include "nnacl/fp32/detection_post_process_fp32.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class DetectionPostProcessInt8CPUKernel : public DetectionPostProcessBaseCPUKernel {
 public:
  DetectionPostProcessInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                    const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : DetectionPostProcessBaseCPUKernel(parameter, inputs, outputs, ctx) {}
  ~DetectionPostProcessInt8CPUKernel() = default;

  int8_t *data_int8_ = nullptr;
  float *data_fp32_ = nullptr;
  lite::QuantArg quant_param_;
  int quant_size_ = 0;
  int thread_n_stride_ = 0;
  int DequantizeInt8ToFp32(const int task_id);

 private:
  int Dequantize(lite::Tensor *tensor, float **data);
  int GetInputData();
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_DETECTION_POST_PROCESS_INT8_H_
