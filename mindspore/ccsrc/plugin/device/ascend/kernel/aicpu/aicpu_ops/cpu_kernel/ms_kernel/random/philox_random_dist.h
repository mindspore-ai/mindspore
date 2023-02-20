/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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

#ifndef AI_CPU_PHILOX_RANDOM_DIS_H
#define AI_CPU_PHILOX_RANDOM_DIS_H

#include <algorithm>
#include "utils.h"
#include "cpu_ops_kernel.h"
#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "utils/philox_random.h"
namespace aicpu {
namespace random {
template <class Distribution, typename T1, typename T2>
void FillTaskScalarInput(Distribution dist, PhiloxRandom gen, T1 *input, T2 *output, int64_t output_size,
                         int32_t group_size, bool *ptr_flag, int64_t start_group, int64_t limit_group) {
  gen.Skip(static_cast<uint64_t>(start_group));
  int64_t offset = start_group * group_size;
  int64_t full_group = std::min(limit_group, output_size / group_size);
  for (int64_t index = start_group; index < full_group; index++) {
    dist(&gen, input, output + offset, group_size, ptr_flag);
    offset += group_size;
  }
  if (full_group < limit_group) {
    int64_t remaining_size = output_size - full_group * group_size;
    dist(&gen, input, output + offset, remaining_size, ptr_flag);
  }
}

template <class Distribution, typename T1, typename T2>
void FillTaskTensorInput(Distribution dist, PhiloxRandom gen, T1 *input, T2 *output, int64_t output_size,
                         int32_t group_size, bool *ptr_flag, int64_t start_group, int64_t limit_group) {
  gen.Skip(static_cast<uint64_t>(start_group));
  int64_t offset = start_group * group_size;
  int64_t full_group = std::min(limit_group, output_size / group_size);
  for (int64_t index = start_group; index < full_group; index++) {
    dist(&gen, input + offset, output + offset, group_size, ptr_flag);
    offset += group_size;
  }
  if (full_group < limit_group) {
    int64_t remaining_size = output_size - full_group * group_size;
    dist(&gen, input + offset, output + offset, remaining_size, ptr_flag);
  }
}

template <typename Distribution>
class PhiloxRandomDist {
 public:
  PhiloxRandomDist(int64_t seed, int64_t offset, int64_t parallelLimit) : generator_(seed, offset) {
    kParallelDataNumSameShape_ = parallelLimit;
  }

  template <typename T1, typename T2>
  uint32_t DistCompute(const CpuKernelContext &ctx, T1 *input, T2 *output, int64_t input_size, int64_t output_size) {
    auto group_size = Distribution::kResultElementCount;
    if (group_size <= 0) {
      KERNEL_LOG_ERROR("group_size must greater 0,and group_size are [%ld] ", group_size);
      return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
    }
    auto group_count = (output_size + group_size - 1) / group_size;
    auto gen = generator_;
    bool invalid_flag = true;
    bool *ptr_flag = &invalid_flag;

    if (input == nullptr || input_size <= 1) {
      if (output_size >= kParallelDataNumSameShape_) {
        uint32_t minCoreNum = 1;
        uint32_t validAicpuNum = CpuKernelUtils::GetCPUNum(ctx);
        if (validAicpuNum > kResvCpuNum) {
          validAicpuNum -= kResvCpuNum;
        }
        int64_t maxCoreNum = std::max(minCoreNum, validAicpuNum);
        auto shard = [&gen, output_size, group_size, input, output, ptr_flag](int64_t start_group,
                                                                              int64_t limit_group) {
          FillTaskScalarInput(Distribution(), gen, input, output, output_size, group_size, ptr_flag, start_group,
                              limit_group);
        };

        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, group_count, group_count / maxCoreNum, shard),
                            "PhiloxRandomDist parallelFor failed.");
      } else {
        FillTaskScalarInput(Distribution(), gen, input, output, output_size, group_size, ptr_flag, 0, group_count);
      }
    } else {
      if (output_size >= kParallelDataNumSameShape_) {
        uint32_t minCoreNum = 1;
        uint32_t validAicpuNum = CpuKernelUtils::GetCPUNum(ctx);
        if (validAicpuNum > kResvCpuNum) {
          validAicpuNum -= kResvCpuNum;
        }
        int64_t maxCoreNum = std::max(minCoreNum, validAicpuNum);
        auto shard = [&gen, output_size, group_size, input, output, ptr_flag](int64_t start_group,
                                                                              int64_t limit_group) {
          FillTaskTensorInput(Distribution(), gen, input, output, output_size, group_size, ptr_flag, start_group,
                              limit_group);
        };

        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, group_count, group_count / maxCoreNum, shard),
                            "PhiloxRandomDist parallelFor failed.");
      } else {
        FillTaskTensorInput(Distribution(), gen, input, output, output_size, group_size, ptr_flag, 0, group_count);
      }
    }

    if (invalid_flag == false) {
      KERNEL_LOG_ERROR("input prob is invalid, must be in [0, 1]");
      return KERNEL_STATUS_PARAM_INVALID;
    }

    return KERNEL_STATUS_OK;
  }

 private:
  int64_t kParallelDataNumSameShape_;
  PhiloxRandom generator_;
};
}  // namespace random
}  // namespace aicpu
#endif