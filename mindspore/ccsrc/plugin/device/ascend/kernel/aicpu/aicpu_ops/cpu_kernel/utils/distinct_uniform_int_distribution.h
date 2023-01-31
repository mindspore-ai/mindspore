/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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

#ifndef AICPU_UTILS_DISTINCT_UNIFORM_INT_DISTRIBUTION_H
#define AICPU_UTILS_DISTINCT_UNIFORM_INT_DISTRIBUTION_H

#include <random>
#include <unordered_set>
#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/kernel_log.h"

namespace aicpu {
template <typename IntType = int>
class DistinctUniformIntDistribution {
 public:
  using ResultType = IntType;

 private:
  using SetType = std::unordered_set<ResultType>;
  using DistrType = std::uniform_int_distribution<ResultType>;

 public:
  DistinctUniformIntDistribution(ResultType inf, ResultType sup)
      : inf_(inf), sup_(sup), range_(sup_ - inf_ + 1), distr_(inf_, sup_) {}
  ~DistinctUniformIntDistribution() = default;
  void Reset() {
    uset_.clear();
    distr_.reset();
  }

  template <typename Generator>
  ResultType exec(Generator &engine) {
    if (not(uset_.size() < range_)) {
      std::terminate();
    }
    ResultType res;
    do {
      res = distr_(engine);
    } while (uset_.count(res) > 0);
    uset_.insert(res);
    return res;
  }

 private:
  const ResultType inf_;
  const ResultType sup_;
  const size_t range_ = 0;
  DistrType distr_;
  SetType uset_;
};
}  // namespace aicpu

#endif  // AICPU_UTILS_DISTINCT_UNIFORM_INT_DISTRIBUTION_H_
