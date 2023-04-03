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

#ifndef AICPU_LOG_UNIFORM_CANDIDATE_SAMPLER_KERNELS_H
#define AICPU_LOG_UNIFORM_CANDIDATE_SAMPLER_KERNELS_H

#include <utility>
#include "common/kernel_base.h"
#include "common/kernel_errcode.h"
#include "common/range_sampler.h"
#include "common/kernel_log.h"
#include "proto/node_def.pb.h"
#include "proto/attr.pb.h"

namespace aicpu {
class CandidateSamplerKernel : public KernelBase {
 public:
  explicit CandidateSamplerKernel(const std::string &kernel_name) : KernelBase(kernel_name){};
  ~CandidateSamplerKernel() = default;

 protected:
  uint32_t ParseKernelParam() override;
  template <class RangeSamplerType, typename T>
  uint32_t DoComputeForEachType();
  template <class RangeSamplerType>
  uint32_t CandidateSamplerCompute();

 private:
  int num_true_;
  int num_sampled_;
  bool unique_;
  int64_t range_max_;
  int64_t seed_;
  std::unique_ptr<RangeSampler> sampler_;

  int batch_size_ = 0;
  std::vector<int64_t> x_shape_;

  ::aicpuops::DataType x_dtype_ = ::aicpuops::DataType::MS_UNKNOWN;
  ::aicpuops::DataType true_expected_count_dtype_ = ::aicpuops::DataType::MS_UNKNOWN;

  void set_sampler(RangeSampler *sampler) { sampler_.reset(sampler); }

};  // CandidateSamplerKernel

class LogUniformCandidateSamplerKernel : public CandidateSamplerKernel {
 public:
  explicit LogUniformCandidateSamplerKernel() : CandidateSamplerKernel("LogUniformCandidateSampler"){};
  ~LogUniformCandidateSamplerKernel() = default;

 protected:
  uint32_t DoCompute() override;
};

class UniformCandidateSamplerKernel : public CandidateSamplerKernel {
 public:
  explicit UniformCandidateSamplerKernel() : CandidateSamplerKernel("UniformCandidateSampler"){};
  ~UniformCandidateSamplerKernel() = default;

 protected:
  uint32_t DoCompute() override;
};

}  // namespace aicpu
#endif  // AICPU_LOG_UNIFORM_CANDIDATE_SAMPLER_KERNELS_H
