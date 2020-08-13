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

#include <float.h>
#include <vector>
#include "src/lite_kernel.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/arm/nnacl/op_base.h"
#include "src/runtime/kernel/arm/nnacl/winograd_transform.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"
#include "src/runtime/kernel/arm/base/layout_transform.h"
#include "src/runtime/kernel/arm/nnacl/fp32/conv.h"
#include "src/runtime/kernel/arm/nnacl/fp32/common_func.h"
#include "src/runtime/kernel/arm/nnacl/matmul_parameter.h"
#include "src/runtime/kernel/arm/nnacl/fp32/matmul.h"

namespace mindspore::kernel {
class Convolution1x1CPUKernel : public ConvolutionBaseCPUKernel {
 public:
  Convolution1x1CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                          const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                          const lite::Primitive *primitive)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx, primitive) {
    matmul_param_ = new MatMulParameter();
  }
  ~Convolution1x1CPUKernel();
  int Init() override;
  int Run() override;
  int ReSize() override;

 public:
  int DoConv1x1(int task_id);

 private:
  int InitConv1x1Param();
  int InitConv1x1BiasWeight();
  void InitConv1x1MatmulParam();
  void Pre1x1Trans(float *src_input, float *src_output);

 private:
  MatMulParameter *matmul_param_ = nullptr;
  bool pre_trans_input_ = false;
  int thread_count_ = 0;
  int thread_stride_ = 0;
  float *weight_ptr_ = nullptr;
  float *pack_input_ = nullptr;
  float *input_ptr_ = nullptr;
  float *output_ptr_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_1X1_H_
