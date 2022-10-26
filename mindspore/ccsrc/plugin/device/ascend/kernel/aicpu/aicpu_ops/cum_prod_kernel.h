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
#ifndef AICPU_OPS_AICPU_CUM_PROD_KERNELS_H_
#define AICPU_OPS_AICPU_CUM_PROD_KERNELS_H_

#include <vector>
#include "common/kernel_base.h"

namespace aicpu {
class CumProdKernel : public KernelBase {
 public:
  CumProdKernel() : KernelBase("CumProdKernel") {}
  ~CumProdKernel() = default;

 protected:
  uint32_t ParseKernelParam() override;
  uint32_t DoCompute() override;

  template <typename T>
  uint32_t CumProdTask();

  template <typename T>
  void LaunchCumProd(const T *input, T *output) const;

  template <typename T>
  void CumProd(const T *input, T *output) const;

  template <typename T>
  void CumProdKernelReverse(const T *input, T *output) const;

  template <typename T>
  void LeftMove(const T *input, T *output) const;

  template <typename T>
  void RightMove(const T *input, T *output) const;

  void Reshape();

  aicpuops::DataType dtype_{aicpuops::DataType::MS_UNKNOWN};
  std::vector<int64_t> shape_;
  std::vector<int64_t> dst_shape_;
  size_t stride_{0};
  size_t stride2_{0};
  size_t dims_[3]{1};
  bool exclusive_{false};
  bool reverse_{false};
  size_t axis_{0};
  size_t lens_{1};
};
}  // namespace aicpu
#endif  // AICPU_OPS_AICPU_CUM_PROD_KERNELS_H_
