/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
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

#ifndef AICPU_UtILS_SPARSE_DENSE_CWISE_UTILS_H_
#define AICPU_UtILS_SPARSE_DENSE_CWISE_UTILS_H_

#include <string>
#include "context/inc/cpu_kernel_utils.h"
#include "inc/ms_cpu_kernel.h"
#include "utils/bcast.h"
#include "utils/eigen_tensor.h"

namespace aicpu {
struct AddOp {
  static std::string Name() { return "Add"; }
};

struct DivOp {
  static std::string Name() { return "Div"; }
};

struct MulOp {
  static std::string Name() { return "Mul"; }
};

template <typename Op>
class SparseDenseCwiseOpKernel : public CpuKernel {
 public:
  SparseDenseCwiseOpKernel() = default;
  ~SparseDenseCwiseOpKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override = 0;

  static uint32_t CheckParams(CpuKernelContext &ctx);

  template <typename T>
  uint32_t SparseDenseCwiseOpSpecialCompute(BcastShapeType type, CpuKernelContext &ctx);
  template <typename T>
  uint32_t SparseDenseCwiseOpSpecialComputeComplex(BcastShapeType type, CpuKernelContext &ctx);

  template <typename T>
  uint32_t ComputeOp(CpuKernelContext &ctx);

  template <typename T>
  uint32_t ComputeOpComplex(CpuKernelContext &ctx);

  template <typename T>
  uint32_t SparseDenseCwiseOpNoBcastCompute(CpuKernelContext &ctx);

  template <typename T>
  uint32_t SparseDenseCwiseOpNoBcastComputeComplex(CpuKernelContext &ctx);

  template <typename T>
  uint32_t SparseDenseCwiseOpBcastCompute(CpuKernelContext &ctx);

  template <typename T>
  uint32_t SparseDenseCwiseOpBcastComputeComplex(CpuKernelContext &ctx);

  template <typename T>
  uint32_t SparseDenseCwiseOpCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
