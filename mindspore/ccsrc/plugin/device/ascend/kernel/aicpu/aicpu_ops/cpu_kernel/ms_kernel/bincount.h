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

#ifndef AICPU_KERNELS_NORMALIZED_BINCOUNT_H_
#define AICPU_KERNELS_NORMALIZED_BINCOUNT_H_

#include <map>

#include "inc/ms_cpu_kernel.h"

namespace aicpu {
class BincountCpuKernel : public CpuKernel {
 public:
  BincountCpuKernel() = default;
  ~BincountCpuKernel() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  static const uint32_t kThirdInputIndex = 2;
  void SetMap();

  std::map<int, std::map<int, std::function<void(Tensor *, int32_t, Tensor *, Tensor *, CpuKernelContext &)>>> calls_;
};
}  // namespace aicpu
#endif
