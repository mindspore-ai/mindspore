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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_CONVOLUTION_WINOGRAD_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_CONVOLUTION_WINOGRAD_FP16_H_

#include <arm_neon.h>
#include <vector>
#include "src/litert/lite_kernel.h"
#include "src/litert/kernel/cpu/base/convolution_base.h"
#include "nnacl/fp16/conv_fp16.h"
#include "nnacl/fp16/winograd_utils_fp16.h"
#include "src/common/utils.h"
#include "nnacl/base/minimal_filtering_generator.h"

namespace mindspore::kernel {
class ConvolutionWinogradFP16CPUKernel : public ConvolutionBaseCPUKernel {
 public:
  ConvolutionWinogradFP16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                   const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx, int out_unit,
                                   void *origin_weight, void *origin_bias)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx, origin_weight, origin_bias), output_unit_(out_unit) {}
  ~ConvolutionWinogradFP16CPUKernel() override {}

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id);
  int InitTmpBuffer();
  int ConfigInputOutput();
  int WinogradFilterTransformFp16(const float16_t *weight_data, const float *matrix_g, const float *matrix_gt,
                                  int oc_block);
  int AdjustNumberOfThread();

 private:
  int MallocWeightBiasData() override;
  void PackWeight() override;
  void FreeTmpBuffer() {
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
    if (col_buffer_ != nullptr) {
      ctx_->allocator->Free(col_buffer_);
      col_buffer_ = nullptr;
    }
    if (opt_input_trans_ != nullptr) {
      ctx_->allocator->Free(opt_input_trans_);
      opt_input_trans_ = nullptr;
    }
  }
  int FilterWeight();
  int kernel_unit_ = 0;
  int input_unit_ = 0;
  int output_unit_;
  float16_t *tmp_data_ = nullptr;
  float16_t *trans_input_ = nullptr;
  float16_t *gemm_out_ = nullptr;
  float16_t *col_buffer_ = nullptr;
  float16_t *opt_input_trans_ = nullptr;
  float matrix_g_[64];
  float matrix_gt_[64];
  TmpBufferAddressFp16 tmp_buffer_address_list_[5] = {0};
  TransFp16FuncList trans_func_;
  int col_tile_ = 0;
  int row_tile_ = 0;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_CONVOLUTION_WINOGRAD_FP16_H_
