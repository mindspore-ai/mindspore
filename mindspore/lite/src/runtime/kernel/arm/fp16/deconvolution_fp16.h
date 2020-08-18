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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_DECONVOLUTION_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_DECONVOLUTION_H_

#include <float.h>
#include <vector>
#include "src/lite_kernel.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "schema/model_generated.h"
#include "src/runtime/kernel/arm/fp16/convolution_base_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/deconv_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/matmul_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/pack_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/cast_fp16.h"

namespace mindspore::kernel {
class DeConvolutionFp16CPUKernel : public ConvolutionBaseFP16CPUKernel {
 public:
  DeConvolutionFp16CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                             const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                             const mindspore::lite::PrimitiveC *primitive)
      : ConvolutionBaseFP16CPUKernel(parameter, inputs, outputs, ctx, primitive) {
    matmul_param_ = new MatMulParameter();
  }
  ~DeConvolutionFp16CPUKernel() override;
  int Init() override;
  int Run() override;
  int ReSize() override;

 public:
  int DoDeconv(int task_id);

 private:
  void FreeParam();
  int InitParam();
  int InitWeightBias();

 private:
  MatMulParameter *matmul_param_;
  int row16_;
  int col8_;
  int input_plane_;
  int kernel_plane_;
  int output_plane_;
  int thread_count_;
  int thread_stride_;
  float16_t *pack_input_;
  float16_t *pack_output_;
  float16_t *tmp_buffer_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_DECONVOLUTION_H_
