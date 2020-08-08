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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_WINOGRAD_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_WINOGRAD_H_

#include <vector>
#include "src/lite_kernel.h"

#include "src/runtime/kernel/arm/nnacl/winograd_transform.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"
#include "src/runtime/kernel/arm/base/matrix.h"

namespace mindspore::kernel {
class ConvolutionWinogradCPUKernel : public ConvolutionBaseCPUKernel {
 public:
  ConvolutionWinogradCPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                               const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                               const lite::Primitive *primitive, int output_unit)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx, primitive), output_unit_(output_unit) {}
  ~ConvolutionWinogradCPUKernel() override {
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
  };
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
  float *tmp_data_;
  float *trans_input_;
  float *gemm_out_;
  float *tmp_out_data_;
  Matrix *trans_weight_;
  InputTransformUnitFunc input_trans_func_;
  OutputTransformUnitFunc output_trans_func_;
  TmpBufferAddress tmp_buffer_address_list_[5];
  GEMM_FUNC_FP32 gemm_func_ = nullptr;
};
void WinogradFilterTransform(const float *weight_data, Matrix *trans_weight, int kernel_unit, int input_unit,
                             ConvParameter *conv_param, int oc_block);
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_WINOGRAD_H_
