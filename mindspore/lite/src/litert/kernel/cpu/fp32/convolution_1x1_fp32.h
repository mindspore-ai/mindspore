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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_CONVOLUTION_1X1_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_CONVOLUTION_1X1_FP32_H_

#include <float.h>
#include <vector>
#include "src/litert/lite_kernel.h"
#include "include/errorcode.h"
#include "nnacl/op_base.h"
#include "src/litert/kernel/cpu/base/convolution_base.h"
#include "src/litert/kernel/cpu/base/layout_transform.h"
#include "nnacl/base/conv1x1_base.h"
#include "nnacl/fp32/common_func_fp32.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/fp32/matmul_fp32.h"

namespace mindspore::kernel {
class Convolution1x1CPUKernel : public ConvolutionBaseCPUKernel {
 public:
  Convolution1x1CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                          const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                          float *origin_weight, float *origin_bias)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx, origin_weight, origin_bias) {}
  ~Convolution1x1CPUKernel();
  int Prepare() override;
  int Run() override;
  int ReSize() override;

 public:
  int DoConv1x1(int task_id);
  int DoConv1x1Hw(int task_id);

 private:
  int InitConv1x1Param();
  int InitConv1x1MatmulParam();
  int MallocWeightBiasData() override;
  void PackWeight() override;
  void FreeTmpBuffer();
  void PackMatmulInput(const float *src_ptr, float *dst_ptr, int row, int col) const;

 private:
  MatMulParameter *matmul_param_ = nullptr;
  bool pre_trans_input_ = false;
  bool multi_thread_by_hw_ = false;
  int thread_count_ = 0;
  int thread_stride_ = 0;
  float *pack_input_ = nullptr;
  float *input_ptr_ = nullptr;
  float *output_ptr_ = nullptr;
  int row_tile_ = 0;
  int col_tile_ = 0;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_CONVOLUTION_1X1_FP32_H_
