/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_AFFINE_FP32_H
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_AFFINE_FP32_H

#include <vector>
#include "src/inner_kernel.h"
#include "nnacl/affine_parameter.h"
#include "nnacl/splice_parameter.h"

namespace mindspore::kernel {
class AffineFp32CPUKernel : public InnerKernel {
 public:
  AffineFp32CPUKernel(OpParameter *param, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(param, inputs, outputs, ctx) {
    affine_parameter_ = reinterpret_cast<AffineParameter *>(param);
  }
  ~AffineFp32CPUKernel() override;

  int Init() override;
  int ReSize() override;
  int Run() override;

 private:
  int FullRunInit();
  int IncrementInit();
  bool CheckAffineValid();
  int CheckActivationValid();
  kernel::InnerKernel *FullMatmulKernelCreate();
  kernel::InnerKernel *IncrementMatmulKernelCreate();
  OpParameter *MatmulParameterCreate();

  int IncrementSplice();
  int IncrementMatmulRun();
  int FullMatmulRun();
  int FullSpliceRun();
  int DoActivation(lite::Tensor *tensor);

 private:
  AffineParameter *affine_parameter_{nullptr};
  kernel::InnerKernel *full_mult_kernel_{nullptr};
  kernel::InnerKernel *increment_mult_kernel_{nullptr};

  lite::Tensor *full_input_{nullptr};
  lite::Tensor *increment_input_{nullptr};
  lite::Tensor *increment_output_{nullptr};

  lite::Tensor *tensor_read_{nullptr};

  bool full_run_{true};
  int src_to_dst_row_offset_{0};
  int matmul_col_{0};
  int matmul_row_{0};
  int splice_src_row_{0};
  int splice_dst_row_{0};
  int splice_src_col_{0};
  int splice_dst_col_{0};
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_AFFINE_FP32_H
