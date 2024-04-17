/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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
#ifndef AICPU_KERNELS_NORMALIZED_SLICEGRAD_H_
#define AICPU_KERNELS_NORMALIZED_SLICEGRAD_H_

#include "inc/ms_cpu_kernel.h"
#include <vector>

namespace aicpu {
class SliceGradKernel : public CpuKernel {
 public:
  SliceGradKernel() = default;
  ~SliceGradKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T, typename S>
  uint32_t SliceGradTask(CpuKernelContext &ctx);
  bool CheckParams(CpuKernelContext &ctx) const;
  uint32_t ParseKernelParam(CpuKernelContext &ctx);
  bool CheckBeginSizeValue(CpuKernelContext &ctx);

  std::vector<int64_t> dy_shape_;
  std::vector<int64_t> begin_shape_;
  std::vector<int64_t> size_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<int64_t> begin_value_;
  std::vector<int64_t> size_value_;
  DataType dy_type_{DT_UNDEFINED};
  DataType begin_type_{DT_UNDEFINED};
};
}  // namespace aicpu
#endif
