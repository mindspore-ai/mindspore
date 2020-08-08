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
#include "src/runtime/kernel/arm/nnacl/winograd_transform.h"

namespace mindspore::kernel {
class Convolution3x3CPUKernel : public ConvolutionBaseCPUKernel {
 public:
  Convolution3x3CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                          const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                          const lite::Primitive *primitive)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx, primitive) {}
  ~Convolution3x3CPUKernel() override {
    if (transformed_filter_addr_ != nullptr) {
      free(transformed_filter_addr_);
    }
    if (tile_buffer_ != nullptr) {
      free(tile_buffer_);
    }
    if (block_unit_buffer_ != nullptr) {
      free(block_unit_buffer_);
    }
    if (tmp_dst_buffer_ != nullptr) {
      free(tmp_dst_buffer_);
    }
    if (nc4hw4_out_ != nullptr) {
      free(nc4hw4_out_);
    }
  };

  int Init() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id);
  int InitWeightBias();
  int InitTmpBuffer();
  void ConfigInputOutput();

 private:
  float *transformed_filter_addr_;
  float *tile_buffer_;
  float *block_unit_buffer_;
  float *tmp_dst_buffer_;
  float *nc4hw4_out_;
  TmpBufferAddress tmp_buffer_address_list_[4];
  GEMM_FUNC_FP32 gemm_func_ = nullptr;
};
void ProcessFilter(float *origin_weight, float *dst_weight, ConvParameter *conv_param, int oc_block, int oc_block_num);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_3X3_H_
