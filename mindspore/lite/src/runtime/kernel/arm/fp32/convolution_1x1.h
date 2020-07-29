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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_1X1_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_1X1_H_

#include <vector>
#include "src/lite_kernel.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/arm/opclib/op_base.h"
#include "src/runtime/kernel/arm/opclib/winograd_transform.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"
#include "src/runtime/kernel/arm/base/layout_transform.h"
#include "src/runtime/kernel/arm/opclib/fp32/conv.h"
#include "src/runtime/kernel/arm/opclib/fp32/common_func.h"

namespace mindspore::kernel {
class Convolution1x1CPUKernel : public ConvolutionBaseCPUKernel {
 public:
  Convolution1x1CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                          const std::vector<lite::tensor::Tensor *> &outputs, const Context *ctx)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx) {}
  ~Convolution1x1CPUKernel();
  int Init() override;
  int Run() override;
  int ReSize() override;

 public:
  int DoStrassen(int task_id);
  int DoPostFunc(int task_id);

 private:
  int InitConv1x1Param();
  int InitConv1x1BiasWeight();
  void InitConv1x1MatmulParam();
  void Pre1x1Trans(float *src_input, float *src_output);

 private:
  StrassenMatMulParameter *matmul_param_ = nullptr;
  bool pre_trans_input_ = false;
  int thread_count_ = 0;
  int thread_hw_count_ = 0;
  int thread_hw_stride_ = 0;
  int thread_oc4_count_ = 0;
  int thread_oc_stride_ = 0;
  float *bias_ptr_ = nullptr;
  float *weight_ptr_ = nullptr;
  float *tmp_ptr_ = nullptr;
  float *c4_input_ = nullptr;
  float *c4_output_ = nullptr;
  float *input_ptr_ = nullptr;
  float *output_ptr_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_1X1_H_

