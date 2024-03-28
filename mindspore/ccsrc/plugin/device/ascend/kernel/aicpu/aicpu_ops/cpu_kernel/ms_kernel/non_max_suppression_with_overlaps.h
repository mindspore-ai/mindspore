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
#ifndef AICPU_KERNELS_NORMALIZED_NON_MAX_SUPPRESSION_WITH_OVERLAPS_H_
#define AICPU_KERNELS_NORMALIZED_NON_MAX_SUPPRESSION_WITH_OVERLAPS_H_

#include "inc/ms_cpu_kernel.h"
#include "cpu_types.h"
#include "utils/eigen_tensor.h"

namespace aicpu {
class NonMaxSuppressionWithOverlapsCpuKernel : public CpuKernel {
 public:
  ~NonMaxSuppressionWithOverlapsCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t GetInputAndCheck(CpuKernelContext &ctx);
  template <typename T, typename T_threshold>
  uint32_t DoNonMaxSuppressionWithOverlapsOp(CpuKernelContext &ctx);

  const Tensor *overlaps_ = nullptr;
  Tensor *scores_ = nullptr;
  Tensor *overlap_threshold_tensor_ = nullptr;
  Tensor *score_threshold_tensor_ = nullptr;
  Tensor *output_indices_ = nullptr;
  int32_t num_boxes_ = 0;
  int32_t max_output_size_ = 0;
  DataType overlaps_dtype_ = DT_UINT32;
  DataType scores_dtype_ = DT_UINT32;
  DataType overlap_threshold_dtype_ = DT_UINT32;
  DataType score_threshold_dtype_ = DT_UINT32;
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_NON_MAX_SUPPRESSION_WITH_OVERLAPS_H_
