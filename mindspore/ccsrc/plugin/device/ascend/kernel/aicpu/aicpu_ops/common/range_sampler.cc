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

#include "common/range_sampler.h"
#include <cmath>
#include <unordered_set>
#include <vector>
#include <random>
#include "common/distinct_uniform_int_distribution.h"

namespace aicpu {

RangeSampler::~RangeSampler() {}

void RangeSampler::SampleBatch(bool unique, const std::vector<int64_t> &batch) const {}

void RangeSampler::SampleBatchGetExpectedCount(bool unique, int64_t seed, std::vector<int64_t> *batch,
                                               std::vector<float> *batch_expected_count, std::vector<int64_t> extras,
                                               std::vector<float> *extras_expected_count) const {
  SampleBatchGetExpectedCountAvoid(unique, seed, batch, batch_expected_count, extras, extras_expected_count,
                                   std::vector<int64_t>());
}

namespace {
static float ExpectedCountHelper(float p, int batch_size, int num_tries) {
  if (num_tries == batch_size) {
    return p * batch_size;
  }
  return -std::expm1(num_tries * std::log1p(-p));
}

template <class Collection>
bool InsertIfNotPresent(Collection *const collection, const typename Collection::value_type &vt) {
  return collection->insert(vt).second;
}

static const int32_t kint32max = static_cast<int32_t>(0x7FFFFFFF);

}  // namespace

void RangeSampler::SampleBatchGetExpectedCountAvoid(bool unique, int64_t seed, std::vector<int64_t> *batch,
                                                    std::vector<float> *batch_expected_count,
                                                    std::vector<int64_t> extras,
                                                    std::vector<float> *extras_expected_count,
                                                    std::vector<int64_t> avoided_values) const {
  const int batch_size = batch->size();
  int num_tries;
  if (range_ <= 0) {
    AICPU_LOGE("range_ must be greater than 0!");
    return;
  }

  rng_.seed(seed);
  if (unique) {
    if (batch_size + avoided_values.size() > static_cast<size_t>(range_)) {
      AICPU_LOGE("the value should be less than range_: %d, but got %d", range_, batch_size + avoided_values.size());
      return;
    }
    std::unordered_set<int64_t> used(batch_size);
    used.insert(avoided_values.begin(), avoided_values.end());
    int num_picked = 0;
    num_tries = 0;
    while (num_picked < batch_size) {
      num_tries++;
      if (num_tries >= kint32max) {
        AICPU_LOGE("num_tries: %d should be less than kint32max: %d!", num_tries, kint32max);
        return;
      }
      int64_t value = Sample();
      if (InsertIfNotPresent(&used, value)) {
        (*batch)[num_picked++] = value;
      }
    }
  } else {
    if (avoided_values.size() != size_t{0}) {
      AICPU_LOGE("avoided_values only supported with unique=true");
      return;
    }
    for (int i = 0; i < batch_size; i++) {
      (*batch)[i] = Sample();
    }
    num_tries = batch_size;
  }

  if (!batch_expected_count->empty()) {
    if (batch_size != static_cast<int>(batch_expected_count->size())) {
      AICPU_LOGE("the size of extras_expected_count: %zu should be equal to batch_size: %d!",
                 batch_expected_count->size(), batch_size);
      return;
    }
    for (int i = 0; i < batch_size; i++) {
      (*batch_expected_count)[i] = ExpectedCountHelper(Probability((*batch)[i]), batch_size, num_tries);
    }
  }
  if (extras.size() != extras_expected_count->size()) {
    AICPU_LOGE("the size of extras and extras_expected_count should be equal!");
    return;
  }
  for (size_t i = 0; i < extras.size(); i++) {
    (*extras_expected_count)[i] = ExpectedCountHelper(Probability(extras[i]), batch_size, num_tries);
  }
}

UniformSampler::UniformSampler(int64_t range) : RangeSampler(range), inv_range_(1.0 / range) {}

int64_t UniformSampler::Sample() const {
  aicpu::distinct_uniform_int_distribution<> dis(0, range_ - 1);
  return dis.exec(&rng_);
}

float UniformSampler::Probability(int64_t value) const { return inv_range_; }

LogUniformSampler::LogUniformSampler(int64_t range) : RangeSampler(range), log_range_(log1p(range)) {}

int64_t LogUniformSampler::Sample() const {
  std::uniform_real_distribution<float> uni_real(0.0, 1.0);

  const int64_t value = static_cast<int64_t>(exp(uni_real(rng_) * log_range_)) - 1;
  if (value < 0) {
    AICPU_LOGE("value: %d should be >= 0", value);
    return 0;
  }

  return value % range_;
}

float LogUniformSampler::Probability(int64_t value) const { return (log((value + 2.0) / (value + 1.0))) / log_range_; }
}  // namespace aicpu
