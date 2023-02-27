/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_MATMUL_FP32_SSE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_MATMUL_FP32_SSE_H_

#if defined(ENABLE_SSE)
#include <vector>
#include "src/litert/kernel/cpu/fp32/matmul_fp32_base.h"
namespace mindspore::kernel {
class MatmulFp32SSECPUKernel : public MatmulFp32BaseCPUKernel {
 public:
  MatmulFp32SSECPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs, const mindspore::lite::InnerContext *ctx)
      : MatmulFp32BaseCPUKernel(parameter, inputs, outputs, ctx) {
    params_->matmul_type_ = kNotImplemented;
  }
  ~MatmulFp32SSECPUKernel() = default;

  void InitGlobalVariable() override;
  int PackMatrixAImplOpt() override;
  int ParallelRunByBatch(int task_id) const override;
  int ParallelRunByRow(int task_id) const override;
  int ParallelRunByOC(int task_id) const override;
  bool CheckThreadCuttingByRow() override;
};
}  // namespace mindspore::kernel
#endif

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_MATMUL_FP32_SSE_H_
