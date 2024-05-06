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

#ifndef AICPU_AICPU_OPS_UNIFORM_CANDIDATE_SAMPLER_H
#define AICPU_AICPU_OPS_UNIFORM_CANDIDATE_SAMPLER_H

#include "inc/ms_cpu_kernel.h"

#include <random>
#include <utility>
#include <string>
#include <vector>
#include <memory>

#include "utils/range_sampler.h"

namespace aicpu {
class CandidateSamplerKernel : public CpuKernel {
 public:
  explicit CandidateSamplerKernel() = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t ParseKernelParam(CpuKernelContext &ctx);
  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);

  int num_true_;
  int num_sampled_;
  bool unique_;
  int64_t range_max_;
  int64_t seed_;
  std::mt19937 rng_;
  std::unique_ptr<RangeSampler> sampler_;

  int batch_size_;
  std::vector<int64_t> x_shape_;
  std::string kernel_name_;

  DataType x_dtype_;
  DataType true_expected_count_dtype_;

  void set_sampler(RangeSampler *sampler) { sampler_.reset(sampler); }
};  // CandidateSamplerKernel
}  // namespace aicpu
#endif  // AICPU_AICPU_OPS_CANDIDATE_SAMPLER_KERNELS_H
