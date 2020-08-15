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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_3x3_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_3x3_FP16_H_

#include <arm_neon.h>
#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/fp16/convolution_base_fp16.h"
#include "src/runtime/kernel/arm/nnacl/optimized_kernel.h"

namespace mindspore::kernel {
class Convolution3x3FP16CPUKernel : public ConvolutionBaseFP16CPUKernel {
 public:
  Convolution3x3FP16CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                              const std::vector<lite::tensor::Tensor *> &outputs, const Context *ctx,
                              const lite::Primitive *primitive)
      : ConvolutionBaseFP16CPUKernel(parameter, inputs, outputs, ctx, primitive) {}
  ~Convolution3x3FP16CPUKernel() override {
    if (fp16_input_ != nullptr) {
      free(fp16_input_);
    }
    if (fp16_weight_ != nullptr) {
      free(fp16_weight_);
    }
    if (fp16_out_ != nullptr) {
      free(fp16_out_);
    }
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
    if (tmp_out_ != nullptr) {
      free(tmp_out_);
    }
  }

  int Init() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id);
  int InitWeightBias();
  int InitTmpBuffer();
  void ConfigInputOutput();

 private:
  float16_t *transformed_filter_addr_;
  float16_t *tile_buffer_;
  float16_t *block_unit_buffer_;
  float16_t *tmp_dst_buffer_;
  float16_t *tmp_out_;
};
void ProcessFilterFp16(float16_t *origin_weight, float16_t *dst_weight, ConvParameter *conv_param);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_3x3_FP16_H_
