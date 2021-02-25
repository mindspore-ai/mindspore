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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_1x1_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_1x1_FP16_H_

#include <arm_neon.h>
#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/fp16/convolution_base_fp16.h"
#include "src/common/utils.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/fp16/matmul_fp16.h"

namespace mindspore::kernel {
class Convolution1x1FP16CPUKernel : public ConvolutionBaseFP16CPUKernel {
 public:
  Convolution1x1FP16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                              const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx, void *origin_weight,
                              void *origin_bias, TypeId origin_weight_data_type, TypeId origin_bias_data_type)
      : ConvolutionBaseFP16CPUKernel(parameter, inputs, outputs, ctx, origin_weight_data_type, origin_bias_data_type),
        origin_weight_(origin_weight),
        origin_bias_(origin_bias) {}
  ~Convolution1x1FP16CPUKernel() override;

  int Init() override;
  int ReSize() override;
  int Run() override;

 public:
  int RunOc(int task_id);
  int RunHw(int task_id);

 private:
  void FreeTmpBuffer();
  int InitConv1x1Param();
  int InitMatmulParam();
  int InitWeightBias();

 private:
  bool pre_trans_input_ = false;
  bool multi_thread_by_hw_ = false;
  int thread_count_ = 1;
  int thread_stride_ = 0;
  void *origin_weight_;  // do not free
  void *origin_bias_;    // do not free
  float16_t *weight_ptr_ = nullptr;
  float16_t *input_ptr_ = nullptr;
  float16_t *pack_input_ = nullptr;
  float16_t *output_ptr_ = nullptr;
  MatMulParameter *matmul_param_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_1x1_FP16_H_
