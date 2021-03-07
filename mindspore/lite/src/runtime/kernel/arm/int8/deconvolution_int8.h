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
#include "nnacl/matmul_parameter.h"
#include "nnacl/int8/deconv_int8.h"
#include "nnacl/int8/common_func_int8.h"
#include "nnacl/int8/matmul_int8.h"
#include "src/runtime/kernel/arm/base/layout_transform.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"

namespace mindspore::kernel {
class DeConvInt8CPUKernel : public ConvolutionBaseCPUKernel {
 public:
  DeConvInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx) {}
  ~DeConvInt8CPUKernel() override;

  int ReSize() override;
  int Init() override;
  int Run() override;

 public:
  int DoDeconv(int task_id);

 private:
  void FreeTmpBuffer();
  int InitData();
  int InitParam();
  int InitBiasWeight();
  void CheckSupportOptimize();
  int InitRunBuf();
  void FreeRunBuf();

 private:
  int32_t *tmp_buffer_ = nullptr; /* record matmul result */
  int32_t *tmp_output_ = nullptr; /* record post c8 result */
  int32_t *input_sum_ = nullptr;  /* record in * w_zp  */
  int32_t *weight_sum_ = nullptr; /* record w_v * in_zp - in_zp * w_zp */
  int8_t *input_ptr_ = nullptr;   /* packed input */
  int8_t *weight_ptr_ = nullptr;  /* packed weight */
  int8_t *output_ptr_ = nullptr;
  size_t thread_count_ = 1;
  size_t thread_stride_ = 0;
  MATMUL_OPT_R4_FUNC matmul_func_;
  MatMulParameter *matmul_param_ = nullptr;
  bool support_optimize_ = true;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_DECONVOLUTION_INT8_H_
