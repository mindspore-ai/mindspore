/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <random>
#include <vector>
#include <iostream>
#include "common/common_test.h"
#include "include/utils/random.h"

namespace mindspore {
class TestRandom : public UT::Common {};

/// Feature: Philox random number generator.
/// Description: Test Philox random number generator.
/// Expectation: Random number generator works as expected.
TEST_F(TestRandom, test_philox_generator) {
  const uint64_t seed = 1234;
  auto rng = random::Philox(seed);
  std::vector<uint32_t> numbers;
  for (size_t i = 0; i < 20; ++i) {
    numbers.push_back(rng());
  }

  // Discard.
  rng = random::Philox(seed);
  rng.discard(0);
  EXPECT_EQ(rng(), numbers[0]);

  rng = random::Philox(seed);
  rng.discard(8);
  EXPECT_EQ(rng(), numbers[8]);

  rng = random::Philox(seed);
  rng.discard(9);
  EXPECT_EQ(rng(), numbers[9]);

  rng = random::Philox(seed);
  rng.discard(10);
  EXPECT_EQ(rng(), numbers[10]);

  rng = random::Philox(seed);
  rng.discard(11);
  EXPECT_EQ(rng(), numbers[11]);

  rng = random::Philox(seed);
  rng.discard(12);
  EXPECT_EQ(rng(), numbers[12]);

  rng = random::Philox(seed);
  rng.discard(13);
  EXPECT_EQ(rng(), numbers[13]);

  // Discard after generate.
  rng = random::Philox(seed);
  rng();
  rng.discard(8 - 1);
  EXPECT_EQ(rng(), numbers[8]);
  rng.discard(1);
  EXPECT_EQ(rng(), numbers[10]);

  rng = random::Philox(seed);
  rng();
  rng.discard(9 - 1);
  EXPECT_EQ(rng(), numbers[9]);
  rng.discard(2);
  EXPECT_EQ(rng(), numbers[12]);

  rng = random::Philox(seed);
  rng();
  rng.discard(10 - 1);
  EXPECT_EQ(rng(), numbers[10]);
  rng.discard(3);
  EXPECT_EQ(rng(), numbers[14]);

  rng = random::Philox(seed);
  rng();
  rng.discard(11 - 1);
  EXPECT_EQ(rng(), numbers[11]);
  rng.discard(4);
  EXPECT_EQ(rng(), numbers[16]);

  rng = random::Philox(seed);
  rng();
  rng.discard(12 - 1);
  EXPECT_EQ(rng(), numbers[12]);
  rng.discard(5);
  EXPECT_EQ(rng(), numbers[18]);

  rng = random::Philox(seed);
  rng();
  rng.discard(13 - 1);
  EXPECT_EQ(rng(), numbers[13]);
}

/// Feature: Random distributions.
/// Description: Test random distributions.
/// Expectation: distributions works as expected.
TEST_F(TestRandom, test_distributions) {
  using Rng = random::Philox;
  using Uniform = random::UniformDistribution<float>;
  using Normal = random::NormalDistribution<float>;
  using TruncNormal = random::TruncatedNormal<float>;

  const uint64_t seed = 4321;
  const size_t length = 10000;
  const size_t half_len = length / 2;
  std::vector<float> randoms1(length);
  std::vector<float> randoms2(length);

  random::GenerateRandoms<float, Rng, Uniform>(seed, 0, randoms1.data(), length, 0.0f, 1.0f);
  random::GenerateRandoms<float, Rng, Uniform>(seed, 0, randoms2.data(), half_len, 0.0f, 1.0f);
  random::GenerateRandoms<float, Rng, Uniform>(seed, half_len, randoms2.data() + half_len, half_len, 0.0f, 1.0f);
  EXPECT_EQ(randoms1, randoms2);

  random::GenerateRandoms<float, Rng, Normal>(seed, 0, randoms1.data(), length, 0.0f, 1.0f);
  random::GenerateRandoms<float, Rng, Normal>(seed, 0, randoms2.data(), half_len, 0.0f, 1.0f);
  random::GenerateRandoms<float, Rng, Normal>(seed, half_len, randoms2.data() + half_len, half_len, 0.0f, 1.0f);
  EXPECT_EQ(randoms1, randoms2);

  random::GenerateRandoms<float, Rng, TruncNormal>(seed, 0, randoms1.data(), length, -2.0f, 2.0f, 0.0f, 1.0f);
  random::GenerateRandoms<float, Rng, TruncNormal>(seed, 0, randoms2.data(), half_len, -2.0f, 2.0f, 0.0f, 1.0f);
  random::GenerateRandoms<float, Rng, TruncNormal>(seed, half_len, randoms2.data() + half_len, half_len, -2.0f, 2.0f,
                                                   0.0f, 1.0f);
  EXPECT_EQ(randoms1, randoms2);
}

/// Feature: Parallel task size compute.
/// Description: Test parallel task size compute.
/// Expectation: Result parallel task size is correct.
TEST_F(TestRandom, test_parallel_task_size) {
  auto result = random::ComputeTaskNumSize(1000, 10);
  EXPECT_EQ(result.first, 1);
  EXPECT_EQ(result.second, 1000);

  result = random::ComputeTaskNumSize(1024, 10);
  EXPECT_EQ(result.first, 1);
  EXPECT_EQ(result.second, 1024);

  result = random::ComputeTaskNumSize(1025, 10);
  EXPECT_EQ(result.first, 10);
  EXPECT_EQ(result.second, 104);

  result = random::ComputeTaskNumSize(2020, 10);
  EXPECT_EQ(result.first, 10);
  EXPECT_EQ(result.second, 204);

  result = random::ComputeTaskNumSize(2021, 10);
  EXPECT_EQ(result.first, 10);
  EXPECT_EQ(result.second, 204);

  result = random::ComputeTaskNumSize(2040, 10);
  EXPECT_EQ(result.first, 10);
  EXPECT_EQ(result.second, 204);

  result = random::ComputeTaskNumSize(2041, 10);
  EXPECT_EQ(result.first, 10);
  EXPECT_EQ(result.second, 208);
}
}  // namespace mindspore
