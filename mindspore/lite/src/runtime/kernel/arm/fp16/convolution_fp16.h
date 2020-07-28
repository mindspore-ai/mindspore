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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_FP16_H_

#include <arm_neon.h>
#include <vector>
#include "src/lite_kernel.h"

#include "src/runtime/kernel/arm/base/convolution_base.h"
#include "src/runtime/kernel/arm/opclib/optimized_kernel.h"

namespace mindspore::kernel {
typedef void (*FP16_GEMM_FUNC)(float16_t *output, float16_t *input, float16_t *weight, float16_t *bias, size_t step,
                               size_t ic4, size_t oc8, size_t offset, size_t mode, size_t writeC4, size_t relu,
                               size_t relu6);

class ConvolutionFP16CPUKernel : public ConvolutionBaseCPUKernel {
 public:
  ConvolutionFP16CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                           const std::vector<lite::tensor::Tensor *> &outputs, const Context *ctx)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx) {}
  ~ConvolutionFP16CPUKernel() override {
    if (fp16_input_ != nullptr) {
      free(fp16_input_);
    }
    if (fp16_weight_ != nullptr) {
      free(fp16_weight_);
    }
    if (packed_input_ != nullptr) {
      free(packed_input_);
    }
    if (packed_weight_ != nullptr) {
      free(packed_weight_);
    }
    if (tmp_output_block_ != nullptr) {
      free(tmp_output_block_);
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
  bool support_fp16_ = true;
  float16_t *fp16_input_;
  float16_t *fp16_weight_;
  float16_t *packed_input_;
  float16_t *packed_weight_;
  float16_t *tmp_output_block_;
  FP16_GEMM_FUNC gemm_func_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_FP16_H_

