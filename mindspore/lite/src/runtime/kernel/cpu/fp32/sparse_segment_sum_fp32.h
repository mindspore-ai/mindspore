/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_SPARSE_SEGMENT_SUM_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_SPARSE_SEGMENT_SUM_FP32_H_

#include <vector>
#include "src/runtime/lite_kernel.h"
#include "include/context.h"
// #include "nnacl/fp32/sparse_segment_sum_fp32.h"
// #include "src/runtime/kernel/cpu/base/layout_transform.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class SparseSegmentSumCPUKernel : public LiteKernel {
 public:
  SparseSegmentSumCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                            const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    // sparse_segment_sum_param_ = reinterpret_cast<SparseSegmentSumParameter *>(op_parameter_);
  }
  ~SparseSegmentSumCPUKernel() = default;

  int PreProcess() override;
  int Prepare() override;
  int ReSize() override { return RET_OK; }
  int Run() override;
  int RunInferOutDataShape();
  int RunSparseSegmentSumCalc();
  // virtual int DoExcute(int task_id);

 protected:
  // SparseSegmentSumParameter *sparse_segment_sum_param_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_SPARSE_SEGMENT_SUM_FP32_H_
