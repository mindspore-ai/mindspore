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
#include "nnacl/fp16/conv_fp16.h"
#include "nnacl/fp16/winograd_utils_fp16.h"
#include "nnacl/optimized_kernel.h"
#include "nnacl/minimal_filtering_generator.h"

namespace mindspore::kernel {
class ConvolutionWinogradFP16CPUKernel : public ConvolutionBaseFP16CPUKernel {
 public:
  ConvolutionWinogradFP16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                   const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx,
                                   const mindspore::lite::PrimitiveC *primitive, int out_unit)
      : ConvolutionBaseFP16CPUKernel(parameter, inputs, outputs, ctx, primitive), output_unit_(out_unit) {}
  ~ConvolutionWinogradFP16CPUKernel() override {
    if (fp16_weight_ != nullptr) {
      free(fp16_weight_);
      fp16_weight_ = nullptr;
    }
    if (trans_weight_ != nullptr) {
      free(trans_weight_);
      trans_weight_ = nullptr;
    }
    FreeTransformMatrices();
  }

  int Init() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id);
  int InitWeightBias();
  int MallocTransformMatrices();
  void FreeTransformMatrices();
  int InitTmpBuffer();
  int ConfigInputOutput();
  int PostProcess();
  int WinogradFilterTransformFp16(const float16_t *weight_data, float *matrix_g, float *matrix_gt, int oc_block);

 private:
  void FreeTmpBuffer() {
    if (nhwc4_input_ != nullptr) {
      ctx_->allocator->Free(nhwc4_input_);
      nhwc4_input_ = nullptr;
    }
    if (trans_input_ != nullptr) {
      ctx_->allocator->Free(trans_input_);
      trans_input_ = nullptr;
    }
    if (tmp_data_ != nullptr) {
      ctx_->allocator->Free(tmp_data_);
      tmp_data_ = nullptr;
    }
    if (gemm_out_ != nullptr) {
      ctx_->allocator->Free(gemm_out_);
      gemm_out_ = nullptr;
    }
    if (tmp_out_data_ != nullptr) {
      ctx_->allocator->Free(tmp_out_data_);
      tmp_out_data_ = nullptr;
    }
  }
  int kernel_unit_;
  int input_unit_;
  int output_unit_;
  float16_t *tmp_data_ = nullptr;
  float16_t *trans_input_ = nullptr;
  float16_t *gemm_out_ = nullptr;
  float16_t *tmp_out_data_ = nullptr;
  float16_t *matrix_a_ = nullptr;
  float16_t *matrix_at_ = nullptr;
  float16_t *matrix_b_ = nullptr;
  float16_t *matrix_bt_ = nullptr;
  float16_t *trans_weight_ = nullptr;
  TmpBufferAddressFp16 tmp_buffer_address_list_[4];
  MatricesFp16 matrices_[4];
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_WINOGRAD_FP16_H_
