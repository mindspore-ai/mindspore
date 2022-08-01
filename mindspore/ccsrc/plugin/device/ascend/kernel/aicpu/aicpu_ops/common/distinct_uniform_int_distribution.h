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
#ifndef AICPU_OPS_AICPU_DISTINCT_UNIFORM_INT_DISTRIBUTION_H_
#define AICPU_OPS_AICPU_DISTINCT_UNIFORM_INT_DISTRIBUTION_H_

#include <random>
#include <unordered_set>

namespace aicpu {
template <typename IntType = int>
class distinct_uniform_int_distribution {
 public:
  using result_type = IntType;

 private:
  using set_type = std::unordered_set<result_type>;
  using distr_type = std::uniform_int_distribution<result_type>;

 public:
  distinct_uniform_int_distribution(result_type inf, result_type sup)
      : inf_(inf), sup_(sup), range_(sup_ - inf_ + 1), distr_(inf_, sup_) {}
  ~distinct_uniform_int_distribution() = default;
  void reset() {
    uset_.clear();
    distr_.reset();
  }

  template <typename Generator>
  result_type exec(Generator *engine) {
    if (!(uset_.size() < range_)) {
      std::terminate();
    }
    result_type res;
    do {
      res = distr_(*engine);
    } while (uset_.count(res) > 0);
    (void)uset_.insert(res);
    return res;
  }

 private:
  const result_type inf_;
  const result_type sup_;
  const size_t range_;
  distr_type distr_;
  set_type uset_;
};
}  // namespace aicpu
#endif  // AICPU_OPS_AICPU_DISTINCT_UNIFORM_INT_DISTRIBUTION_H_
