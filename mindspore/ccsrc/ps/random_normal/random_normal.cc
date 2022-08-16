/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ps/random_normal/random_normal.h"
#include <random>
#include "include/common/random.h"
#include "utils/log_adapter.h"

namespace mindspore::ps {
bool InitRandomNormal(float mean, float stddev, std::vector<size_t> out_shape, size_t global_seed, size_t op_seed,
                      float *output_data) {
  // Check output data pointer.
  if (output_data == nullptr) {
    MS_LOG(ERROR) << "output data is null.";
    return false;
  }
  // Check shape.
  if (out_shape.size() == 0) {
    MS_LOG(ERROR) << "output data shape is empty.";
    return false;
  }
  // Calculate data size from shape.
  size_t data_size = 1;
  for (size_t i = 0; i < out_shape.size(); ++i) {
    data_size *= out_shape[i];
  }
  // Generate randoms parallel.
  constexpr int seed_shift = 32;
  const uint64_t seed = (global_seed << seed_shift) + op_seed;
  using Generator = random::Philox;
  using Distribution = random::NormalDistribution<float>;
  random::GenerateRandomsParallel<float, Generator, Distribution>(seed, output_data, data_size, mean, stddev);
  return true;
}
}  // namespace mindspore::ps
