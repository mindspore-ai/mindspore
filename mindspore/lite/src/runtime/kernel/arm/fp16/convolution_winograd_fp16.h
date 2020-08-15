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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_WINOGRAD_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_WINOGRAD_FP16_H_

#include <arm_neon.h>
#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/fp16/convolution_base_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/conv_fp16.h"
#include "src/runtime/kernel/arm/fp16/matrix_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/winograd_utils_fp16.h"
#include "src/runtime/kernel/arm/nnacl/optimized_kernel.h"

namespace mindspore::kernel {
class ConvolutionWinogradFP16CPUKernel : public ConvolutionBaseFP16CPUKernel {
 public:
  ConvolutionWinogradFP16CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                                   const std::vector<lite::tensor::Tensor *> &outputs, const Context *ctx,
                                   const lite::Primitive *primitive)
      : ConvolutionBaseFP16CPUKernel(parameter, inputs, outputs, ctx, primitive) {}
  ~ConvolutionWinogradFP16CPUKernel() override {
    if (fp16_input_ != nullptr) {
      free(fp16_input_);
    }
    if (fp16_weight_ != nullptr) {
      free(fp16_weight_);
    }
    if (fp16_out_ != nullptr) {
      free(fp16_out_);
    }
    if (tmp_data_ != nullptr) {
      free(tmp_data_);
    }
    if (trans_input_ != nullptr) {
      free(trans_input_);
    }
    if (gemm_out_ != nullptr) {
      free(gemm_out_);
    }
    if (tmp_out_data_ != nullptr) {
      free(tmp_out_data_);
    }
    delete trans_weight_;
  }

  int Init() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id);
  int InitWeightBias();
  int MallocFilterMatrix(int oc_block, int oc_block_num);
  int InitTmpBuffer();
  int ConfigInputOutput();

 private:
  int kernel_unit_;
  int input_unit_;
  int output_unit_;
  float16_t *tmp_data_;
  float16_t *trans_input_;
  float16_t *gemm_out_;
  float16_t *tmp_out_data_;
  Matrix *trans_weight_;
  InputTransformUnitFp16Func input_trans_func_;
  OutputTransformUnitFp16Func output_trans_func_;
  TmpBufferAddressFp16 tmp_buffer_address_list_[4];
};
void WinogradFilterTransformFp16(const float16_t *weight_data, Matrix *trans_weight, int kernel_unit, int input_unit,
                                 ConvParameter *conv_param, int oc_block);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_WINOGRAD_FP16_H_
