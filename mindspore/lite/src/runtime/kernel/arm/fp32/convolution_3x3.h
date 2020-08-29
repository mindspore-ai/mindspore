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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_3X3_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_3X3_H_

#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"
#include "nnacl/winograd_transform.h"

namespace mindspore::kernel {
class Convolution3x3CPUKernel : public ConvolutionBaseCPUKernel {
 public:
  Convolution3x3CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                          const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                          const mindspore::lite::PrimitiveC *primitive)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx, primitive) {}
  ~Convolution3x3CPUKernel() override {
    if (transformed_filter_addr_ != nullptr) {
      free(transformed_filter_addr_);
    }
  }
  int Init() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id);
  int InitWeightBias();
  int InitTmpBuffer();
  void ConfigInputOutput();
  int PostProcess();

 private:
  void FreeTmpBuffer() {
    if (tile_buffer_ != nullptr) {
      ctx_->allocator->Free(tile_buffer_);
      tile_buffer_ = nullptr;
    }
    if (block_unit_buffer_ != nullptr) {
      ctx_->allocator->Free(block_unit_buffer_);
      block_unit_buffer_ = nullptr;
    }
    if (tmp_dst_buffer_ != nullptr) {
      ctx_->allocator->Free(tmp_dst_buffer_);
      tmp_dst_buffer_ = nullptr;
    }
    if (nc4hw4_out_ != nullptr) {
      ctx_->allocator->Free(nc4hw4_out_);
      nc4hw4_out_ = nullptr;
    }
    if (col_buffer_ != nullptr) {
      ctx_->allocator->Free(col_buffer_);
      col_buffer_ = nullptr;
    }
  }

  float *transformed_filter_addr_ = nullptr;
  float *tile_buffer_ = nullptr;
  float *block_unit_buffer_ = nullptr;
  float *tmp_dst_buffer_ = nullptr;
  float *col_buffer_ = nullptr;
  float *nc4hw4_out_ = nullptr;
  TmpBufferAddress tmp_buffer_address_list_[5];
  GEMM_FUNC_FP32 gemm_func_ = nullptr;
};
void ProcessFilter(float *origin_weight, float *dst_weight, ConvParameter *conv_param, int oc_block, int oc_block_num);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_3X3_H_
