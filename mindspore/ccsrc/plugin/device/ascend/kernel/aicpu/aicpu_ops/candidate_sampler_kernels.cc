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

#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/candidate_sampler_kernels.h"
#include <algorithm>
#include "common/range_sampler.h"
#include "common/random_utils.h"

namespace aicpu {
namespace {
const uint32_t kCountsIndex = 1;
const uint32_t kStatesIndex = 2;
}  // namespace

uint32_t CandidateSamplerKernel::ParseKernelParam() {
  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();
  num_true_ = attrs["num_true"].i();
  num_sampled_ = attrs["num_sampled"].i();
  unique_ = attrs["unique"].b();
  range_max_ = attrs["range_max"].i();
  seed_ = attrs["seed"].i();

  // input0: true_classes
  ::aicpuops::Tensor x_tensor = node_def_.inputs(0);
  x_dtype_ = static_cast<::aicpuops::DataType>(x_tensor.tensor_type());
  const ::aicpuops::TensorShape &x_shape = x_tensor.tensor_shape();
  for (auto i = 0; i < x_shape.dim_size(); i++) {
    x_shape_.emplace_back(x_shape.dim(i).size());
  }

  if (x_shape_.size() != 2) {
    AICPU_LOGE("true_classes must be a matrix");
    return kAicpuKernelStateFailed;
  }
  if (x_shape_[1] != num_true_) {
    AICPU_LOGE(
      "true_classes must have "
      "num_true columns, expected: ",
      x_shape_[1], " was: ", num_true_);
    return kAicpuKernelStateFailed;
  }

  batch_size_ = x_shape.dim(0).size();
  if (x_dtype_ != ::aicpuops::DataType::MS_INT64 && x_dtype_ != ::aicpuops::DataType::MS_INT32) {
    AICPU_LOGE("invalid type of x_dtype_: %d", x_dtype_);
    return kAicpuKernelStateFailed;
  }

  // output_2: sampled_candidates
  ::aicpuops::Tensor true_expected_count_tensor = node_def_.outputs(1);
  true_expected_count_dtype_ = static_cast<::aicpuops::DataType>(true_expected_count_tensor.tensor_type());
  if (true_expected_count_dtype_ != ::aicpuops::DataType::MS_FLOAT32) {
    AICPU_LOGE("invalid type of true_expected_count_dtype_: %d", true_expected_count_dtype_);
    return kAicpuKernelStateFailed;
  }
  return kAicpuKernelStateSucess;
}

template <class RangeSamplerType, typename T>
uint32_t CandidateSamplerKernel::DoComputeForEachType() {
  const int64_t batch_size = x_shape_[0];
  // input
  T *true_classes = reinterpret_cast<T *>(io_addrs_[0]);
  std::vector<T> true_candidate_raw(true_classes, true_classes + batch_size * num_true_);
  std::vector<int64_t> true_candidate(true_candidate_raw.size());
  std::transform(true_candidate_raw.begin(), true_candidate_raw.end(), true_candidate.begin(),
                 [&](T x) { return static_cast<int64_t>(x); });
  std::vector<int64_t> sampled_candidate(num_sampled_);
  std::vector<T> sampled_candidate_raw(num_sampled_);
  std::vector<float> true_expected_count(batch_size * num_true_);
  std::vector<float> sampled_expected_count(num_sampled_);

  // get random generator seed
  uint32_t kernel_ret = 0;
  uint64_t rng_seed = random::GetKernelBaseRandomStates(io_addrs_, kCountsIndex, kStatesIndex,
                                                        static_cast<uint64_t>(seed_), 0, kernel_name_, &kernel_ret);
  if (kernel_ret != kAicpuKernelStateSucess) {
    return kAicpuKernelStateFailed;
  }

  set_sampler(new RangeSamplerType(range_max_));

  if (unique_ && num_sampled_ > sampler_->range()) {
    AICPU_LOGE("For AICPU ops ", kernel_name_, ", the sampler's range is too small.");
    return kAicpuKernelStateFailed;
  }

  sampler_->SampleBatchGetExpectedCount(unique_, rng_seed, &sampled_candidate, &sampled_expected_count, true_candidate,
                                        &true_expected_count);

  std::transform(sampled_candidate.begin(), sampled_candidate.end(), sampled_candidate_raw.begin(),
                 [&](int64_t x) { return static_cast<T>(x); });
  int true_count_size = batch_size * num_true_ * sizeof(float);
  int ret1 =
    memcpy_s(reinterpret_cast<void *>(io_addrs_[3]), num_sampled_ * sizeof(T),
             reinterpret_cast<void *>(&sampled_candidate_raw.front()), sampled_candidate_raw.size() * sizeof(T));
  int ret2 = memcpy_s(reinterpret_cast<void *>(io_addrs_[4]), true_count_size,
                      reinterpret_cast<void *>(&true_expected_count.front()), true_count_size);
  int ret3 =
    memcpy_s(reinterpret_cast<void *>(io_addrs_[5]), num_sampled_ * sizeof(float),
             reinterpret_cast<void *>(&sampled_expected_count.front()), sampled_expected_count.size() * sizeof(float));
  if (ret1 < 0 || ret2 < 0 || ret3 < 0) {
    AICPU_LOGE("memcpy_s failed!");
    return kAicpuKernelStateFailed;
  }

  return kAicpuKernelStateSucess;
}

template <class RangeSamplerType>
uint32_t CandidateSamplerKernel::CandidateSamplerCompute() {
  switch (x_dtype_) {
    case ::aicpuops::DataType::MS_INT32: {
      DoComputeForEachType<RangeSamplerType, int>();
      break;
    }
    case ::aicpuops::DataType::MS_INT64: {
      DoComputeForEachType<RangeSamplerType, int64_t>();
      break;
    }
    default: {
      AICPU_LOGE("CandidateSampler op doesn't support input tensor types.");
      return kAicpuKernelStateFailed;
    }
  }
  return kAicpuKernelStateSucess;
}

uint32_t LogUniformCandidateSamplerKernel::DoCompute() {
  LogUniformCandidateSamplerKernel::CandidateSamplerCompute<LogUniformSampler>();
  return kAicpuKernelStateSucess;
}

uint32_t UniformCandidateSamplerKernel::DoCompute() {
  UniformCandidateSamplerKernel::CandidateSamplerCompute<UniformSampler>();
  return kAicpuKernelStateSucess;
}

}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t LogUniformCandidateSampler(void *param) {
  aicpu::LogUniformCandidateSamplerKernel logUniformCandidateSampler;
  return logUniformCandidateSampler.Compute(param);
}
}

extern "C" {
__attribute__((visibility("default"))) uint32_t UniformCandidateSampler(void *param) {
  aicpu::UniformCandidateSamplerKernel uniformCandidateSampler;
  return uniformCandidateSampler.Compute(param);
}
}
