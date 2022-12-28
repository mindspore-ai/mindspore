/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#ifndef AICPU_KERNELS_NORMALIZED_NMS_WITH_MASK_H
#define AICPU_KERNELS_NORMALIZED_NMS_WITH_MASK_H

#include <vector>
#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "cpu_ops_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_log.h"
#include "securec.h"

namespace aicpu {
class NMSWithMaskCpuKernel : public CpuKernel {
 public:
  NMSWithMaskCpuKernel() = default;
  ~NMSWithMaskCpuKernel() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);

  int num_input_{0};
  float iou_value_{0.0};
  size_t ceil_power_2{0};
  int box_size_ = 5;  //  pre_defined box width
  enum output_list_ { OUTPUT, SEL_IDX, SEL_BOXES };
};
}  // namespace aicpu
#endif
