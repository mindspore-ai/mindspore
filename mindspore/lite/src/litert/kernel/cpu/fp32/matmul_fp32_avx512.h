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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_MATMUL_FP32_AVX512_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_MATMUL_FP32_AVX512_H_

#ifdef ENABLE_AVX512
#include <vector>
#include "src/litert/kernel/cpu/fp32/matmul_fp32_base.h"
namespace mindspore::kernel {
struct MatmulSlice {
  int row_s_ = 0;
  int row_e_ = 0;
  int col_s_ = 0;
  int col_e_ = 0;
};

class MatmulFp32AVX512CPUKernel : public MatmulFp32BaseCPUKernel {
 public:
  MatmulFp32AVX512CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                            const std::vector<lite::Tensor *> &outputs, const mindspore::lite::InnerContext *ctx)
      : MatmulFp32BaseCPUKernel(parameter, inputs, outputs, ctx) {
    params_->matmul_type_ = kNotImplemented;
  }
  ~MatmulFp32AVX512CPUKernel() = default;

  void InitGlobalVariable() override;
  int InitParameter() override;
  int GetThreadCuttingPolicy() override;
  int PackMatrixAImplOpt() override;
  int ParallelRunByBatch(int task_id) const override;
  int ParallelRunByRow(int task_id) const override;
  int ParallelRunByOC(int task_id) const override;
  int ParallelRunByGEMM(int task_id) const override;
  int ParallelRunByBatchColRowGEMM(int task_id) const override;
  int ParallelRunByRow1Deep1GEPDOT(int task_id) const override;
  int ParallelRunByGEPDOT(int task_id) const override;
  int ParallelRunByGEPM(int task_id) const override;
  void BatchRowThreadCut();
  void BatchColThreadCut();
  void BatchColRowSliceThreadCut();
  void BatchColRowThreadCut();
  bool CheckThreadCuttingByRow() override;
  bool SupportMulBatchCuttingByRow() { return true; }

  std::vector<std::vector<MatmulSlice>> matmul_slice_set_;
};
}  // namespace mindspore::kernel
#endif

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_MATMUL_FP32_AVX512_H_
