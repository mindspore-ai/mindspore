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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_DECONVOLUTION_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_DECONVOLUTION_INT8_H_

#include <vector>
#include "src/lite_kernel.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/arm/nnacl/matmul_parameter.h"
#include "src/runtime/kernel/arm/nnacl/int8/deconv.h"
#include "src/runtime/kernel/arm/nnacl/int8/matmul_int8.h"
#include "src/runtime/kernel/arm/base/layout_transform.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"
#include "src/runtime/kernel/arm/nnacl/arithmetic_common.h"

namespace mindspore::kernel {
class DeConvInt8CPUKernel : public ConvolutionBaseCPUKernel {
 public:
  DeConvInt8CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                      const std::vector<lite::tensor::Tensor *> &outputs, const Context *ctx,
                      const lite::Primitive *primitive)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx, primitive) {}
  ~DeConvInt8CPUKernel() override;

  int ReSize() override;
  int Init() override;
  int Run() override;

 public:
  int DoDeconv(int task_id);
  int DoPostFunc(int task_id);

 private:
  int InitData();
  int InitParam();
  int InitBiasWeight();

 private:
  MatMulParameter *fc_param_;
  int8_t *weight_ptr_;
  int8_t *input_ptr_;   /* record c8 input*/
  int32_t *tmp_buffer_; /* record matmul result */
  int32_t *tmp_output_; /* record post c8 result */
  int8_t *output_ptr_;
  size_t thread_count_;
  size_t thread_stride_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_DECONVOLUTION_INT8_H_
