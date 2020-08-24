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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_CONVOLUTION_1x1_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_CONVOLUTION_1x1_INT8_H_

#include <vector>
#include "src/lite_kernel.h"
#include "include/errorcode.h"
#include "schema/model_generated.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"
#include "nnacl/int8/conv_int8.h"
#include "nnacl/int8/matmul_int8.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/optimized_kernel.h"

namespace mindspore::kernel {
class Convolution1x1Int8CPUKernel : public ConvolutionBaseCPUKernel {
 public:
  Convolution1x1Int8CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                              const std::vector<lite::tensor::Tensor *> &outputs, const Context *ctx,
                              const mindspore::lite::PrimitiveC *primitive)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx, primitive) {}
  ~Convolution1x1Int8CPUKernel() override;

  int Init() override;
  int ReSize() override;
  int Run() override;

 public:
  int RunImpl(int task_id);

 private:
  void FreeResizeBuf();
  int InitParam();
  int InitWeightBias();
  void Pre1x1Trans(int8_t *src_input, int8_t *src_output);
  void CheckSupportOptimize();

 private:
  int32_t *input_sum_ = nullptr; /* per-channel: oc4 format */
  int8_t *packed_weight_ = nullptr;
  int8_t *packed_input_ = nullptr;
  int8_t *input_ptr_ = nullptr;
  int8_t *output_ptr_ = nullptr;
  size_t thread_count_ = 1;
  size_t thread_stride_ = 0;
  bool pre_trans_input_ = false;
  MatMulParameter *matmul_param_ = nullptr;
  MATMUL_OPT_R_FUNC matmul_func_ = nullptr;
  bool support_optimize_ = false;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_CONVOLUTION_1x1_INT8_H_
