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
#ifndef AICPU_OPS_QUANT_DTYPE_CAST_KERNEL_H_
#define AICPU_OPS_QUANT_DTYPE_CAST_KERNEL_H_

#include <vector>
#include <random>
#include "common/kernel_base.h"

namespace aicpu {
class QuantDTypeCastKernel : public KernelBase {
 public:
  QuantDTypeCastKernel() : KernelBase("QuantDTypeCastKernel") {}
  ~QuantDTypeCastKernel() = default;

 protected:
  uint32_t ParseKernelParam() override;
  uint32_t DoCompute() override;
  bool CheckParams() const;
  uint32_t QuantDTypeCastTask();
  void FixedBitHalfDequantTask();
  void FixedBitFloatDequantTask();
  std::vector<int64_t> input_shapes_;
  int64_t quant_param_size_{0};

  int64_t axis_{0};
  int64_t dst_type_{0};
  int64_t src_type_{0};
};
}  // namespace aicpu
#endif  // AICPU_OPS_QUANT_DTYPE_CAST_KERNEL_H_
